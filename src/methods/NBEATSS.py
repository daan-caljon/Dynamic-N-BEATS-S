from torch import nn
import torch
import torch.nn.functional as F


class GenericNBeatsBlock(nn.Module):
    
    def __init__(self,
                 device,
                 backcast_length,
                 forecast_length,
                 hidden_layer_units, thetas_dims, 
                 share_thetas,
                 dropout = False, dropout_p = 0.0, 
                 neg_slope = 0.00):
        
        super().__init__()
        self.device = device
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length        
        if isinstance(hidden_layer_units, int):
            self.hidden_layer_units = [hidden_layer_units for FC_layer in range(4)]
        else:
            #assert(len(hidden_layer_units) == 4)
            self.hidden_layer_units = hidden_layer_units
        self.thetas_dims = thetas_dims
        self.share_thetas = share_thetas
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.neg_slope = neg_slope
        
        # shared layers in block
        self.fc1 = nn.Linear(self.backcast_length,
                             self.hidden_layer_units[0])#, bias = False)
        self.fc2 = nn.Linear(self.hidden_layer_units[0], self.hidden_layer_units[1])#, bias = False)
        self.fc3 = nn.Linear(self.hidden_layer_units[1], self.hidden_layer_units[2])#, bias = False)
        self.fc4 = nn.Linear(self.hidden_layer_units[2], self.hidden_layer_units[3])#, bias = False)
        
        # do not use F.dropout as you want dropout to only affect training (not evaluation mode)
        # nn.Dropout handles this automatically
        if self.dropout:
            self.dropoutlayer = nn.Dropout(p = self.dropout_p)
        
        # task specific (backcast & forecast) layers in block
        # do not include bias - see section 3.1 - Ruben does include bias for generic blocks
        if self.share_thetas:
            self.theta_b_fc = self.theta_f_fc = nn.Linear(self.hidden_layer_units[3], self.thetas_dims)#, bias = False)
        else:
            self.theta_b_fc = nn.Linear(self.hidden_layer_units[3], self.thetas_dims)#, bias = False)
            self.theta_f_fc = nn.Linear(self.hidden_layer_units[3], self.thetas_dims)#, bias = False)
        
        # block output layers
        self.backcast_out = nn.Linear(self.thetas_dims, self.backcast_length)#, bias = False) # include bias - see section 3.3
        self.forecast_out = nn.Linear(self.thetas_dims, self.forecast_length)#, bias = False) # include bias - see section 3.3
        
        
    def forward(self, x):
        
        if self.dropout:
            h1 = F.leaky_relu(self.fc1(x.to(self.device)), negative_slope = self.neg_slope)
            h1 = self.dropoutlayer(h1)
            h2 = F.leaky_relu(self.fc2(h1), negative_slope = self.neg_slope)
            h2 = self.dropoutlayer(h2)
            h3 = F.leaky_relu(self.fc3(h2), negative_slope = self.neg_slope)
            h3 = self.dropoutlayer(h3)
            h4 = F.leaky_relu(self.fc4(h3), negative_slope = self.neg_slope)
            theta_b = F.leaky_relu(self.theta_b_fc(h4), negative_slope = self.neg_slope)
            #theta_b = self.theta_b_fc(h4)
            theta_f = F.leaky_relu(self.theta_f_fc(h4), negative_slope = self.neg_slope)
            #theta_f = self.theta_f_fc(h4)
            backcast = self.backcast_out(theta_b)
            forecast = self.forecast_out(theta_f)
        else:
            h1 = F.leaky_relu(self.fc1(x.to(self.device)), negative_slope = self.neg_slope)
            h2 = F.leaky_relu(self.fc2(h1), negative_slope = self.neg_slope)
            h3 = F.leaky_relu(self.fc3(h2), negative_slope = self.neg_slope)
            h4 = F.leaky_relu(self.fc4(h3), negative_slope = self.neg_slope)
            theta_b = F.leaky_relu(self.theta_b_fc(h4), negative_slope = self.neg_slope)
            #theta_b = self.theta_b_fc(h4)
            theta_f = F.leaky_relu(self.theta_f_fc(h4), negative_slope = self.neg_slope)
            #theta_f = self.theta_f_fc(h4)
            backcast = self.backcast_out(theta_b)
            forecast = self.forecast_out(theta_f)
            
        return backcast, forecast
    
    
    def __str__(self):
        
        block_type = type(self).__name__
        
        return f'{block_type}(units={self.hidden_layer_units}, thetas_dims={self.thetas_dims}, ' \
            f'backcast_length={self.backcast_length}, ' \
            f'forecast_length={self.forecast_length}, share_thetas={self.share_thetas}, ' \
            f'dropout={self.dropout}, dropout_p={self.dropout_p}, neg_slope={self.neg_slope}) at @{id(self)}'
    
# Only the forward method is changed compared to standard NBeatsNet
class StableNBeatsNet(nn.Module): 
    
    def __init__(self, 
                 device,
                 backcast_length_multiplier,
                 forecast_length,
                 hidden_layer_units, thetas_dims, 
                 share_thetas,
                 nb_blocks_per_stack, n_stacks, share_weights_in_stack,
                 dropout = False, dropout_p = 0.0, 
                 neg_slope = 0.00):
        
        super().__init__()
        self.device = device
        self.backcast_length = backcast_length_multiplier * forecast_length
        self.forecast_length = forecast_length
        self.hidden_layer_units = hidden_layer_units
        self.thetas_dims = thetas_dims
        self.share_thetas = share_thetas
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.n_stacks = n_stacks
        self.share_weights_in_stack = share_weights_in_stack
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.neg_slope = neg_slope
        
        self.stacks = []
        self.parameters = []
        
        print(f'| N-Beats')
        for stack_id in range(self.n_stacks):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        
        
    def create_stack(self, stack_id):
        
        print(f'| --  Stack Generic (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights
            else:
                block = GenericNBeatsBlock(self.device,
                                           self.backcast_length,
                                           self.forecast_length,
                                           self.hidden_layer_units, self.thetas_dims, 
                                           self.share_thetas,
                                           self.dropout, self.dropout_p, 
                                           self.neg_slope)
                self.parameters.extend(block.parameters())
                print(f'     | -- {block}')
                blocks.append(block)
                
        return blocks

    
    def forward(self, backcast_arr):
        
        # dim backcast_arr = batch_size x shifts x backcast_length
        # shifts == 0 is standard input window, others are shifted lookback windows 
        # higher index = further back in time
        # feed different input windows (per batch) through the SAME network (check via list of learnable parameters)
        # see https://stackoverflow.com/questions/54444630/application-of-nn-linear-layer-in-pytorch-on-additional-dimentions
        
        forecast_arr = torch.zeros((backcast_arr.shape[0], # take batch size from backcast
                                    backcast_arr.shape[1], # take n of shifts from backcast
                                    self.forecast_length), dtype = torch.float).to(self.device)
        backcast_arr = backcast_arr.to(self.device)
        
        # loop through stacks (and blocks)
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast_arr)
                backcast_arr = backcast_arr - b
                forecast_arr = forecast_arr + f  
                
        return backcast_arr, forecast_arr