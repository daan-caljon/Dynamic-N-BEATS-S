import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
import gc
import copy
import wandb
from time import time
import plotly.graph_objects as go
from src.methods.utils.metrics import RMSSE, RMSSE_m, SMAPE, MAPE
# from src.methods.NBEATSS import GenericNBeatsBlock
from src.methods.NBEATSS import StableNBeatsNet



def seed_torch(seed = 5101992):
    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 

class StableNBeatsLearner:
    
    def __init__(self,
                 device,
                 forecast_length,
                 configNBeats):
        
        gc.collect()
                
        self.device = device 
        self.forecast_length = forecast_length
        self.configNBeats = configNBeats
        
        if self.configNBeats["loss_function"] == 1:
            self.loss = RMSSE
        elif self.configNBeats["loss_function"] == 2:
            self.loss = RMSSE_m
        elif self.configNBeats["loss_function"] == 3:
            self.loss = SMAPE
        elif self.configNBeats["loss_function"] == 4:
            self.loss = MAPE
            
        self.rndseed = self.configNBeats["rndseed"]
        seed_torch(self.rndseed)
        
        print('--- Model ---')    
        self.model = StableNBeatsNet(self.device,
                                     self.configNBeats["backcast_length_multiplier"],
                                     self.forecast_length,
                                     self.configNBeats["hidden_layer_units"],
                                     self.configNBeats["thetas_dims"],
                                     self.configNBeats["share_thetas"],
                                     self.configNBeats["nb_blocks_per_stack"],
                                     self.configNBeats["n_stacks"],
                                     self.configNBeats["share_weights_in_stack"],                               
                                     self.configNBeats["dropout"],
                                     self.configNBeats["dropout_p"],
                                     self.configNBeats["neg_slope"])
        
        self.model = self.model.to(self.device)

        #ADDED
        if self.configNBeats["balance_type"] == "uw":
            self.loss_scale = nn.Parameter(torch.tensor([-0.5,-0.5]).to(self.device)) 
            self.optim = torch.optim.Adam(list(self.model.parameters()) + [self.loss_scale], 
                                          lr = self.configNBeats["learning_rate"],
                                          weight_decay = self.configNBeats["weight_decay"])
        else:
            self.optim = torch.optim.Adam(self.model.parameters(), 
                                      lr = self.configNBeats["learning_rate"],
                                      weight_decay = self.configNBeats["weight_decay"])
        
        self.init_state = copy.deepcopy(self.model.state_dict())
        self.init_state_opt = copy.deepcopy(self.optim.state_dict())
        
        wandb.watch(self.model)
    
    
    def ts_padding(self, ts_train_data, ts_eval_data):
        
        # Some time series in the dataset are not long enough to support the specified:
        # forecast_length and backcast_length + backcast input shifts
        # we use zero padding for the time series that are too short (neutral effect on loss calculations)
        # + self.shifts comes from the number of extra observations needed to create the shifted inputs/targets
        # + (self.forgins - 1) comes from rolling origin evaluation
        
        length_train = (self.configNBeats["backcast_length_multiplier"] * self.forecast_length + 
                        self.forecast_length +
                        self.shifts)
        length_eval = (self.configNBeats["backcast_length_multiplier"] * self.forecast_length +
                       self.forecast_length +
                       self.shifts +
                       (self.forigins - 1))
        
        ts_train_pad = [x if x.size >= length_train else np.pad(x,
                                                                (int(length_train - x.size), 0), 
                                                                'constant', 
                                                                constant_values = 0) for x in ts_train_data]
        ts_eval_pad = [x if x.size >= length_eval else np.pad(x,
                                                              (int(length_eval - x.size), 0), 
                                                              'constant', 
                                                              constant_values = 0) for x in ts_eval_data]
        
        return ts_train_pad, ts_eval_pad
        
        
    def make_batch(self, batch_data, shuffle_origin = True):
        
        # If shuffle_origin = True --> batch for training --> random forecast origin based on LH 
        # If shuffle_origin = False --> batch for evaluation --> fixed forecast origin
        
        # Split the batch into input_list and target_list
        # In x_arr and target_arr: batch x shift x backcats_length/forecast_length
        x_arr = np.empty(shape = (len(batch_data), 
                                  self.shifts + 1,
                                  self.configNBeats["backcast_length_multiplier"] * self.forecast_length))
        target_arr = np.empty(shape = (len(batch_data), 
                                       self.shifts + 1, 
                                       self.forecast_length))
        
        # For every time series in the batch:
        # (1) slice the time series according to specific forecasting origin (depending on shuffle_origin)
        # (2) make shifted inputs/targets --> max number of shifts = forecast_length - 1
        # (3) fill x_arr and target_arr
        for j in range(len(batch_data)):
            i = batch_data[j]
            
            if shuffle_origin: 
                # suffle_origin --> only in training 
                
                ### --> also pick random scale --> does not result in improved results
                ### to remain as close as possible to nbeats paper: do not pick random scale
                ### i = i + i * np.random.default_rng().uniform(-0.95, 0.95, 1)
                
                # pick origin
                LH_max_offset = int(self.configNBeats["LH"] * self.forecast_length)
                ts_max_offset = int(len(i) -
                                    (self.configNBeats["backcast_length_multiplier"] * self.forecast_length + 
                                     self.forecast_length +
                                     self.shifts))
                max_offset = min(LH_max_offset, ts_max_offset)
                if max_offset < 1:
                    offset = np.zeros(1)
                else:
                    offset = np.random.randint(low = 0, high = max_offset)   
            else:
                offset = np.zeros(1)
            
            if offset == 0:
                for shift in range(self.shifts + 1):
                    if shift == 0:
                        x_arr[j, shift, :] = i[-self.forecast_length-self.configNBeats["backcast_length_multiplier"]*self.forecast_length:-self.forecast_length]
                        target_arr[j, shift, :] = i[-self.forecast_length:]
                    else:
                        x_arr[j, shift, :] = i[-self.forecast_length-self.configNBeats["backcast_length_multiplier"]*self.forecast_length-shift:-self.forecast_length-shift]
                        target_arr[j, shift, :] = i[-self.forecast_length-shift:-shift]
            else:
                for shift in range(self.shifts + 1):
                    x_arr[j, shift, :] = i[-self.forecast_length-self.configNBeats["backcast_length_multiplier"]*self.forecast_length-offset-shift:-self.forecast_length-offset-shift]
                    target_arr[j, shift, :] = i[-self.forecast_length-offset-shift:-offset-shift]
                    
        return x_arr, target_arr
                                        
                    
    def create_example_plots(self, output, target, actuals_train, final_evaluation = False):
        
        plot_forecasts = torch.cat((actuals_train, output))
        plot_actuals = torch.cat((actuals_train, target))
        random_sample_forecasts = plot_forecasts.squeeze()
        random_sample_actuals = plot_actuals.squeeze()
        x_axis = torch.arange(1, random_sample_forecasts.shape[0]+1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = x_axis.numpy(), y = random_sample_forecasts.numpy(),
                                 mode = 'lines+markers', name = 'forecasts'))
        fig.add_trace(go.Scatter(x = x_axis.numpy(), y = random_sample_actuals.numpy(),
                                 mode = 'lines+markers', name = 'actuals'))
        
        # We only visualize examples for last epoch
        if not final_evaluation:
            wandb.log({"example_plots_evaluation": fig})
        else:
            wandb.log({"example_plots_final_evaluation": fig})
            
            
    def evaluate(self, x_arr, target_arr,
                 epoch = None,
                 need_grad = True,
                 early_stop = False):
        
        losses = dict()
        
        # Inputs must be converted to np.array of Tensors (float)
        x_arr = torch.from_numpy(x_arr).float().to(self.device)
        target_arr = torch.from_numpy(target_arr).float().to(self.device)
        
        if need_grad:
            self.model.train() 
            self.model.to(self.device)
            _, forecast_arr = self.model(x_arr) 
            
            losses_forecast_shifts = 0.0
            for shift in range(self.shifts + 1):
                losses_forecast_shifts += self.loss(forecast_arr[:, shift, :], 
                                                    target_arr[:, shift, :], 
                                                    x_arr[:, shift, :])
            losses["forecast_accuracy"] = losses_forecast_shifts / (self.shifts + 1)
                        
            if self.shifts > 0:
                # dimensions = batch_size x shifted forecasts for stability computations
                forecast_base_arr = torch.zeros((forecast_arr.shape[0],
                                                 sum(range(self.forecast_length - self.shifts, self.forecast_length))),
                                                dtype = torch.float).to(self.device)
                forecast_shift_arr = torch.zeros((forecast_arr.shape[0],
                                                  sum(range(self.forecast_length - self.shifts, self.forecast_length))),
                                                 dtype = torch.float).to(self.device)
                col = 0
                for shift in range(1, self.shifts + 1):
                    for horizon_m1 in range(self.forecast_length - shift):
                        forecast_base_arr[:, col] = forecast_arr[:, 0, horizon_m1]
                        forecast_shift_arr[:, col] = forecast_arr[:, shift, horizon_m1 + shift]
                        col = col + 1
                losses["forecast_stability"] = self.loss(forecast_shift_arr, forecast_base_arr, x_arr[:, 0, :])
            else:
                losses["forecast_stability"] = torch.zeros(1)
                
        else:
            with torch.no_grad():
                
                self.model.eval()
                self.model.to(self.device)
                
                _, forecast_arr = self.model(x_arr) #Dit duurt lang! (wat is het verschil met forward)
                
                losses_forecast_shifts = 0.0
                for shift in range(self.shifts + 1):
                    losses_forecast_shifts += self.loss(forecast_arr[:, shift, :], target_arr[:, shift, :], x_arr[:, shift, :])
                losses["forecast_accuracy"] = losses_forecast_shifts / (self.shifts + 1)
                
                if self.shifts > 0:
                    forecast_base_arr = torch.zeros((forecast_arr.shape[0],
                                                     sum(range(self.forecast_length - self.shifts, self.forecast_length))),
                                                    dtype = torch.float).to(self.device)
                    forecast_shift_arr = torch.zeros((forecast_arr.shape[0],
                                                      sum(range(self.forecast_length - self.shifts, self.forecast_length))),
                                                     dtype = torch.float).to(self.device)
                    col = 0
                    #print("begin forecast stability loop")
                    for shift in range(1, self.shifts + 1):
                        for horizon_m1 in range(self.forecast_length - shift):
                            forecast_base_arr[:, col] = forecast_arr[:, 0, horizon_m1]
                            forecast_shift_arr[:, col] = forecast_arr[:, shift, horizon_m1 + shift]
                            col = col + 1
                    #print("forecast stability klaar")
                    losses["forecast_stability"] = self.loss(forecast_base_arr, forecast_shift_arr, x_arr[:, 0, :])
                else:
                    losses["forecast_stability"] = torch.zeros(1)
                    
                if not self.disable_plot:
                    if early_stop:
                        # Plot validation examples - of standard/unshifted input - for last epoch before break
                        # This part of the evaluation function is only called after training has been forced to stop
                        self.create_example_plots(forecast_arr[0, 0, :], target_arr[0, 0, :], x_arr[0, 0, :])
                    else:
                        # Plot validation examples - of standard/unshifted input - for last epoch
                        # This part of the evaluation function is called after training has been completed
                        if (epoch == self.configNBeats["epochs"]):
                            self.create_example_plots(forecast_arr[0, 0, :], target_arr[0, 0, :], x_arr[0, 0, :])
        
        return losses
    
    
    # Training of net (training data can include validation data) + validation or testing
    def train_net(self,
                  ts_train_m4m,
                  ts_eval_m4m,
                  forigins,
                  validation = True,
                  validation_earlystop = False,
                  disable_plot = True):
        
        self.forigins = forigins
        self.shifts = self.configNBeats["shifts"]
        self.validation = validation
        self.validation_earlystop = validation_earlystop
        self.disable_plot = disable_plot
        #assert self.shifts < self.forecast_length # max allowed number of shifts is forecast_length - 1
        
        # Data preprocessing depends on backcast_length_multiplier
        ts_train_pad, ts_eval_pad = self.ts_padding(ts_train_m4m, ts_eval_m4m)
        ts_train_pad = np.array(ts_train_pad, dtype = object)
        ts_eval_pad = np.array(ts_eval_pad, dtype = object)
        
        print('--- Training ---')
        
        # Containers to save train/evaluation losses and parameters
        tloss_combined, tloss_forecast_accuracy, tloss_forecast_stability = [], [], []
        eloss_combined, eloss_forecast_accuracy, eloss_forecast_stability = [], [], []
        #params = []
        
        # Main training loop
        self.model.load_state_dict(self.init_state)
        self.optim.load_state_dict(self.init_state_opt)
            
        seed_torch(self.rndseed)
        # Initialize early stopping object
        # if self.validation_earlystop:
        #     early_stopping = EarlyStopping(patience = self.configNBeats["patience"], verbose = True)
        cosine = 0
        torch.autograd.set_detect_anomaly(True)

        


        for epoch in range(1, self.configNBeats["epochs"]+1):
            
            start_time = time()
            # Shuffle train data
            np.random.shuffle(ts_train_pad)
            # Determine number of batches per epoch
            num_batches = int(ts_train_pad.shape[0] / self.configNBeats["batch_size"])
            self.mylambda = self.configNBeats["lambda"]
            
            print(epoch,self.mylambda, self.configNBeats["lambda"])    
            ####    
            
            
            # Training per epoch
            avg_tloss_combined_epoch = 0.0
            avg_tloss_forecast_accuracy_epoch = 0.0
            avg_tloss_forecast_stability_epoch = 0.0
            
            for k in range(num_batches):
                print("batch",num_batches)
            
                batch = np.array(ts_train_pad[k*self.configNBeats["batch_size"]:(k+1)*self.configNBeats["batch_size"]])
                x_arr, target_arr = self.make_batch(batch, shuffle_origin = True)
                
                self.optim.zero_grad() #Set gradients to zero
                losses_batch = self.evaluate(x_arr, target_arr,
                                             epoch, need_grad = True, 
                                             early_stop = False)

                
                
                
                if self.configNBeats["balance_type"] == "gcossim" or self.configNBeats["balance_type"] == "weighted gcossim":
              
                    
                    
                    #for name, param in self.model.named_parameters():
                        #if param.requires_grad:
                            #print(name, param.data)
                    main_loss = losses_batch["forecast_accuracy"]
                    aux_loss = losses_batch["forecast_stability"]
                   
                    main_grad = torch.autograd.grad(main_loss,self.model.parameters(),retain_graph=True, allow_unused=True)
                    #One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.
                    # copy
                    my_temp_list = list()
                    for g in main_grad:
                        if g == None:
                            my_temp_list.append(None)
                        else:
                            my_temp_list.append(g.clone())
                    #grad = tuple(my_temp_list)
                    
                    grad = tuple(g for g in my_temp_list)
                    aux_grad = torch.autograd.grad(aux_loss, self.model.parameters(),retain_graph=True, allow_unused=True)

                    my_temp_list = list()
                    for g in main_grad: 
                        if g != None:
                            my_temp_list.append(g)
                    
                    main_grad_flat = torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(my_temp_list)), axis=0)
                    
                    my_temp_list = list()
                    for g in aux_grad:
                        if g != None:
                            my_temp_list.append(g)
                    
                    aux_grad_flat = torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(my_temp_list)), axis=0)
                    
                    cosine = torch.clamp(nn.CosineSimilarity(dim=0)(main_grad_flat, aux_grad_flat), -1, 1)
                    print(cosine, "cosine")
                    
                    if cosine > 0: 
                        my_temp_list = list()
                        for g, ga in zip(grad,aux_grad):
                            if g != None:
                                if self.configNBeats["balance_type"] == "gcossim":
                                    my_temp_list.append(g+ga)
                                elif self.configNBeats["balance_type"] == "weighted gcossim":
                                    my_temp_list.append(g+ga*cosine)
                            else: my_temp_list.append(None)
                        grad = tuple(my_grad for my_grad in my_temp_list)

                        #grad = tuple(g + ga for g, ga in zip(grad, aux_grad))
                    loss_combined = main_loss + aux_loss
                    loss_combined.backward()
                   
                    
                    for p, g in zip(self.model.parameters(), grad):
                        p.grad = g
                        #print(p.grad)
                   
                    self.optim.step()
                    
                    self.optim.zero_grad()
                    del(main_grad,grad,my_temp_list,aux_grad_flat,main_grad_flat,main_loss,aux_loss) #cosine hier weg mss
                    torch.cuda.empty_cache()
                
                #uncertainty waiting adapted from https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/UW.py
                elif self.configNBeats["balance_type"] == "uw":
                    # if epoch == 1:
                        #Here loss_scale is defined as log(sigma**2): loss_scale.exp() = sigma**2 and loss_scale/2 = log(sigma)
                        # self.loss_scale = nn.Parameter(torch.tensor([-0.5,-0.5]).to(self.device))
                        #self.optim.add_param_group({'params': self.loss_scale})
                    weight_acc = 1/(2*self.loss_scale[0].exp())
                    weight_stab = 1/(2*self.loss_scale[1].exp())
                    # #We scale so it sums to 1
                    # weight_acc = weight_acc/(weight_acc + weight_stab)
                    # weight_stab = 1 - weight_acc

                    loss_acc = losses_batch["forecast_accuracy"]*weight_acc
                    loss_stab = losses_batch["forecast_stability"]*weight_stab
                    loss_combined = loss_acc + loss_stab + self.loss_scale[0]/2 + self.loss_scale[1]/2
                    self.optim.zero_grad()
                    loss_combined.backward()
                    self.optim.step()
                    self.mylambda = self.loss_scale[1].exp()
                    print(self.loss_scale)


                elif self.configNBeats["balance_type"] == "rw":
                    torch.autograd.set_detect_anomaly(True)
                    #implementation with softmax:
                    #losses = torch.tensor([losses_batch["forecast_accuracy"],losses_batch["forecast_stability"]])
                    
                    #weights = F.softmax(torch.randn(2), dim = -1).to(self.device) #RLW: create random weights
                    #print(weights)
                    
                    #WITHOUT CAP
                    #weights = torch.rand(2).to(self.device)
                    #weights = weights/weights.sum()
                    
                    weight1 = torch.rand(1).to("cuda")*self.configNBeats["lambda_cap"]
                    weight0 = 1 - weight1
                    weights = torch.cat((weight0, weight1), 0)
                    print(weights)
                    
                    weights_cloned = weights.clone()
                    #loss = torch.mul(losses, weights_cloned).sum()
                    loss_combined = losses_batch["forecast_accuracy"]*weights_cloned[0] + losses_batch["forecast_stability"]*weights_cloned[1]
                    self.optim.zero_grad()
                    loss_combined.backward()
                    self.optim.step()
                    self.mylambda = weights[1]/(weights[0] + weights[1])
                    

                
                elif self.configNBeats["balance_type"] == "gradnorm":
                    #torch.autograd.set_detect_anomaly(True)
                    print(torch.cuda.memory_reserved(device="cuda:0"))
                    losses_batch_tensor = torch.tensor([losses_batch["forecast_accuracy"],losses_batch["forecast_stability"]], device= self.device) #create tensor object
                    
                    if epoch == 1 and k ==0:
                        balance_weights = torch.ones(2, device = self.device) #weights used to balance tasks: weight1*accuracy + weight2*instability
                        balance_weights[0] = 1.9
                        balance_weights[1] = 0.1 #start weights (can be adjusted to get better results)
                        balance_weights = torch.nn.Parameter(balance_weights) #So gradient can be calculated 
                        T = 2 # 2 tasks
                        self.optim_grad = torch.optim.Adam([balance_weights],lr = self.configNBeats["learning_rate_gradnorm"]) #create second optimizer
                        l0 = losses_batch_tensor.detach() #loss in first iteration
                        #get right layer:
                        myblock = self.model.stacks[self.configNBeats["n_stacks"]-1][self.configNBeats["nb_blocks_per_stack"]-1] #last blcok
                        lastlayer = myblock.fc4 #(last shared layer of last block of laatste stack
                    balance_weights_cloned = balance_weights.clone()
                    loss_combined = (balance_weights_cloned[0]*losses_batch["forecast_accuracy"]+balance_weights_cloned[1]*losses_batch["forecast_stability"])
                    loss_combined.backward(retain_graph=True) 
                    gw = []

                    #Calculate the two task gradients with respect to the last shared layer
                    for i in range(0,len(losses_batch_tensor)):  
                        if i ==0:
                            my_temp_loss = "forecast_accuracy" 
                        else:
                            my_temp_loss ="forecast_stability"
                       
                        #task_gradient =  torch.autograd.grad(balance_weights[i]*losses_batch[my_temp_loss], self.model.parameters(), allow_unused = True,retain_graph=True, create_graph=True)[0] 

                        task_gradient =  torch.autograd.grad(balance_weights[i]*losses_batch[my_temp_loss], lastlayer.parameters(), retain_graph=True, create_graph=True)[0] 

                        gw.append(torch.linalg.norm(task_gradient)) #take L2 norm for each task
                    #See definitions GradNorm paper
                    gw = torch.stack(gw) #make it a tensor
                    loss_ratio = losses_batch_tensor.detach() /l0 
                    avg_gw = gw.mean().detach() 
                    rt = loss_ratio/loss_ratio.mean() 
                    desired_grad = (avg_gw*rt**self.configNBeats["alpha"]).detach()
                    lgrad = torch.abs(gw-desired_grad).sum()
                    #clear gradients for the gradnorm optimizer:
                    self.optim_grad.zero_grad()
                    #calculate gradients + do optimization step on weights!
                    lgrad.backward()
                    self.optim_grad.step()
                    #backward pass for weighted task loss:
                    self.optim.step()
                    balance_weights = (balance_weights/balance_weights.sum()*T).detach()
                    print(balance_weights)
                    balance_weights = torch.nn.Parameter(balance_weights)
                    self.optim_grad = torch.optim.Adam([balance_weights], lr= self.configNBeats["learning_rate_gradnorm"])
                    self.mylambda = balance_weights[1]/(balance_weights.sum())
                    del(gw,loss_ratio,rt,desired_grad,lgrad,task_gradient)
                    torch.cuda.empty_cache()
              
                else:    
                    
                    if self.shifts > 0:
                        #loss_combined = ((self.configNBeats["lambda"] * losses_batch["forecast_stability"]) +
                        #((1 - self.configNBeats["lambda"]) * losses_batch["forecast_accuracy"]))
                        loss_combined = ((self.mylambda * losses_batch["forecast_stability"]) +
                        ((1 - self.mylambda) * losses_batch["forecast_accuracy"]))
                    else:
                        loss_combined = losses_batch["forecast_accuracy"]
                
                
                    loss_combined.backward() #calculates gradients based on loss
                    self.optim.step() #optimizer.step is performs a parameter update based on the current gradient 
                    #(stored in .grad attribute of a parameter) and the update rule 
                    #(https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350)
                
                    #params = self.model.parameters()
                    #total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    #if (epoch == 1 or epoch == self.configNBeats["epochs"]) and k == 0:
                    #    print('Epoch {}/{} \t n_learnable_pars={:.4f}'.format(
                    #        epoch,
                    #        self.configNBeats["epochs"],
                    #        total_params))
                
                wandb.log({"tloss_comb_step": loss_combined,
                        "tloss_fcacc_step": losses_batch["forecast_accuracy"],
                        "tloss_fcstab_step": losses_batch["forecast_stability"],
                          "cosine" : cosine,
                          "lambda": self.mylambda})
                #print("wandb logt")
                avg_tloss_combined_epoch += (loss_combined / num_batches)
                avg_tloss_forecast_accuracy_epoch += (losses_batch["forecast_accuracy"] / num_batches)
                avg_tloss_forecast_stability_epoch += (losses_batch["forecast_stability"] / num_batches)
                
            if self.validation: # validation_full and validation_earlystop
               
                # Evaluation per epoch
                avg_eloss_combined_epoch = 0.0
                avg_eloss_forecast_accuracy_epoch = 0.0
                avg_eloss_forecast_stability_epoch = 0.0

                for forigin in range(self.forigins):
                    
                    # Only one batch, but one batch per forecast origin
                    if forigin < self.forigins-1:
                        eval_data_subset = np.array([x[:(-18 + forigin + self.forecast_length)] for x in ts_eval_pad],
                                                   dtype = object)
                    else:
                        eval_data_subset = np.array([x for x in ts_eval_pad], dtype = object)
                    x_arr, target_arr = self.make_batch(eval_data_subset, shuffle_origin = False)
                    
                    losses_evaluation = self.evaluate(x_arr, target_arr,
                                                      epoch, need_grad = False,
                                                      early_stop = False)
                    
                    if self.shifts > 0:
                        #loss_combined = ((self.configNBeats["lambda"] * losses_evaluation["forecast_stability"]) +
                        #                 ((1 - self.configNBeats["lambda"]) * losses_evaluation["forecast_accuracy"]))
                        loss_combined = ((self.mylambda * losses_evaluation["forecast_stability"]) +
                                         ((1 - self.mylambda) * losses_evaluation["forecast_accuracy"]))
                    else:
                        loss_combined = losses_evaluation["forecast_accuracy"]

                    avg_eloss_combined_epoch += (loss_combined / self.forigins)
                    avg_eloss_forecast_accuracy_epoch += (losses_evaluation["forecast_accuracy"] / self.forigins)
                    avg_eloss_forecast_stability_epoch += (losses_evaluation["forecast_stability"] / self.forigins)
                
                elapsed_time = time() - start_time

                print('Epoch {}/{} \t tloss_combined={:.4f} \t eloss_combined={:.4f} \t time={:.2f}s \t lambda={:.2f}'.format(
                    epoch,
                    self.configNBeats["epochs"],
                    avg_tloss_combined_epoch,
                    avg_eloss_combined_epoch,
                    elapsed_time,
                    self.mylambda))
                
                wandb.log({"epoch": epoch,
                           "tloss_comb_evol": avg_tloss_combined_epoch,
                           "tloss_fcacc_evol": avg_tloss_forecast_accuracy_epoch,
                           "tloss_fcstab_evol": avg_tloss_forecast_stability_epoch,
                           "eloss_comb_evol": avg_eloss_combined_epoch,
                           "eloss_fcacc_evol": avg_eloss_forecast_accuracy_epoch,
                           "eloss_fcstab_evol": avg_eloss_forecast_stability_epoch})
                
                
                    
            else: # testing
                
                elapsed_time = time() - start_time

                print('Epoch {}/{} \t tloss_combined={:.4f} \t time={:.2f}s'.format(
                    epoch,
                    self.configNBeats["epochs"],
                    avg_tloss_combined_epoch,
                    elapsed_time))
                
                wandb.log({"epoch": epoch,
                           "tloss_comb_evol": avg_tloss_combined_epoch,
                           "tloss_fcacc_evol": avg_tloss_forecast_accuracy_epoch,
                           "tloss_fcstab_evol": avg_tloss_forecast_stability_epoch})

        wandb.log({"tloss_comb": avg_tloss_combined_epoch,
                   "tloss_fcacc": avg_tloss_forecast_accuracy_epoch,
                   "tloss_fcstab": avg_tloss_forecast_stability_epoch})
        
        print('--- Training done ---')
        print('--- Final evaluation ---')
        
        print('--- M4 evaluation ---')
        
        # Containers to save actuals and forecasts
        actuals = np.empty(shape = (len(ts_eval_pad), self.forigins, self.forecast_length)) # n_series, forigin, forecast_length
        forecasts = np.empty(shape = (len(ts_eval_pad), self.forigins, self.forecast_length)) # n_series, forigin, forecast_length
        
        # Forecasts for each origin in rolling_window
        for forigin in range(self.forigins):
            
            # Only one batch, but one batch per forecast origin
            if forigin < self.forigins-1:
                eval_data_subset = np.array([x[:(-18 + forigin + self.forecast_length)] for x in ts_eval_pad],
                                           dtype = object)
            else:
                eval_data_subset = np.array([x for x in ts_eval_pad], dtype = object)
            x_arr, target_arr = self.make_batch(eval_data_subset, shuffle_origin = False)
            
            # Produce forecasts for subset of test data
            x_arr = torch.from_numpy(x_arr).float().to(self.device)
            target_arr = torch.from_numpy(target_arr).float().to(self.device)
            with torch.no_grad():
                self.model.eval()
                self.model.to(self.device)
                _, forecast_arr = self.model(x_arr)
                
            x_arr = x_arr.cpu() 
            target_arr = target_arr.cpu()
            forecast_arr = forecast_arr.cpu()
                
            # Plot 10 random examples per origin - of standard/unshifted input
            sample_ids = np.random.randint(low = 0, high = int(x_arr.shape[0]), size = 10)
            for sample_id in sample_ids:
                self.create_example_plots(forecast_arr[sample_id, 0, :], 
                                          target_arr[sample_id, 0, :], 
                                          x_arr[sample_id, 0, :],
                                          final_evaluation = True)
                
            # Save to containers
            forecasts[:, forigin, :] = forecast_arr[:, 0, :]
            actuals[:, forigin, :] = target_arr[:, 0, :]
            
        # Compute accuracy sMAPE
        sMAPE = 200 * np.mean(np.abs(actuals - forecasts) / (np.abs(forecasts) + np.abs(actuals)))
        
        # Compute stability Total MAC
        if self.forecast_length == 6:
            weight = np.mean(actuals[:, [0, 6, 12], :], axis = (1, 2))
        elif self.forecast_length == 18:
            weight = np.mean(actuals[:, 0, :], axis = -1)
        forecasts_helper = np.full((actuals.shape[0], 
                                    self.forigins,
                                    (self.forecast_length - 1) + self.forigins), np.nan)
        # n_series x self.forigins x ((forecast_length - 1) + forigins)
        for forigin in range(self.forigins):
            forecasts_helper[:, forigin, forigin:(forigin + self.forecast_length)] = forecasts[:, forigin, :]
        MAC_mat = np.abs(np.diff(forecasts_helper, axis = 1))
        # n_series x (self.forigins - 1) x ((forecast_length - 1) + forigins)
        MAC_mat_adjust = np.delete(MAC_mat, [0, (self.forecast_length - 1) + self.forigins - 1], 2)
        # n_series x (self.forigins - 1) x ((forecast_length - 1) + forigins - 2)
        MAC = np.nanmean(MAC_mat_adjust, axis = 1)
        # n_series x ((forecast_length - 1) + forigins - 2)
        ItemMAC = np.mean(MAC, axis = 1) / weight
        TotalMAC = np.mean(ItemMAC) * 100
        
        print('sMAPE_m4m={:.4f} \t TotalMAC_m4m={:.4f}'.format(sMAPE, TotalMAC))
        
        wandb.log({"sMAPE_m4m": sMAPE,
                   "TotalMAC_m4m": TotalMAC})
        
        # n_series, forigin, forecast_length
        fc_colnames = [str(i) for i in range(1, self.forecast_length + 1)]
        
        actuals_np = actuals#.numpy()
        m,n,r = actuals_np.shape
        actuals_arr = np.column_stack((np.repeat(np.arange(m) + 1, n), 
                                       np.tile(np.arange(n) + 1, m),
                                       actuals_np.reshape(m*n, -1)))
        actuals_df = pd.DataFrame(actuals_arr, columns = ['item_id', 'fc_origin'] + fc_colnames)
        helper_col = ['actual'] * len(actuals_df)
        actuals_df['type'] = helper_col
        
        forecasts_np = forecasts#.numpy()
        m,n,r = forecasts_np.shape
        forecasts_arr = np.column_stack((np.repeat(np.arange(m) + 1, n), 
                                         np.tile(np.arange(n) + 1, m),
                                         forecasts_np.reshape(m*n, -1)))
        forecasts_df = pd.DataFrame(forecasts_arr, columns = ['item_id', 'fc_origin'] + fc_colnames)
        helper_col = ['forecast'] * len(forecasts_df)
        forecasts_df['type'] = helper_col
        
        output_df_m4m = pd.concat([actuals_df, forecasts_df])
         
        wandb.join()
        
        return output_df_m4m