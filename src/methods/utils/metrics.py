# output = batch x fl
# target = batch x fl
# actuals_train = batch x bl

import torch

def SMAPE(output, target, actuals_train = None):
    
    abs_errors = torch.abs(target - output)
    abs_output = torch.abs(output)
    abs_target = torch.abs(target)
    loss = 200 * torch.mean(abs_errors / (abs_output.detach() + abs_target + 1e-5))
    # possibly nan values in training networks if no offset in denominator is used
    
    return loss


def MAPE(output, target, actuals_train = None):

    abs_errors = torch.abs(target - output)
    abs_target = torch.abs(target)
    loss = 100 * torch.mean(abs_errors / (abs_target + 1e-5))
    # possibly nan values in training networks if no offset in denominator is used

    return loss


def MASE(output, target, actuals_train):
    
    mask = torch.abs(actuals_train)>1e-6
    mad = torch.sum(torch.abs(actuals_train[:, 1:] - actuals_train[:, :-1]), dim = -1) / (torch.sum(mask, dim = -1) - 1)
    mad_reshaped = mad.unsqueeze(-1).repeat_interleave(target.shape[-1], dim = -1)
    loss_items = torch.mean((torch.abs(target - output)) / (mad_reshaped + 1e-5), dim = -1)
    loss_items_clamped = torch.clamp(loss_items, 0, 5)
    loss = torch.mean(loss_items_clamped)

    return loss


def MASE_m(output, target, actuals_train):
    
    mask = torch.abs(actuals_train)>1e-6
    mad = torch.sum(torch.abs(actuals_train[:, 12:] - actuals_train[:, :-12]), dim = -1) / (torch.sum(mask, dim = -1) - 12)
    mad_reshaped = mad.unsqueeze(-1).repeat_interleave(target.shape[-1], dim = -1)
    loss_items = torch.mean((torch.abs(target - output)) / (mad_reshaped + 1e-5), dim = -1)
    loss_items_clamped = torch.clamp(loss_items, 0, 5)
    loss = torch.mean(loss_items_clamped) 

    return loss


def RMSSE(output, target, actuals_train):
    
    mask = torch.abs(actuals_train)>1e-6
    msd = torch.sum((actuals_train[:, 1:] - actuals_train[:, :-1])**2, dim = -1) / (torch.sum(mask, dim = -1) - 1)
    msd_reshaped = msd.unsqueeze(-1).repeat_interleave(target.shape[-1], dim = -1)
    loss_items = torch.sqrt(torch.mean((target - output)**2 / (msd_reshaped + 1e-5), dim = -1))
    loss_items_clamped = torch.clamp(loss_items, 0, 5)
    loss = torch.mean(loss_items_clamped)
    #loss = torch.sqrt(torch.mean((target - output)**2 / msd_reshaped))

    return loss


def RMSSE_m(output, target, actuals_train):
    
    mask = torch.abs(actuals_train)>1e-6
    msd = torch.sum((actuals_train[:, 12:] - actuals_train[:, :-12])**2, dim = -1) / (torch.sum(mask, dim = -1) - 12)
    msd_reshaped = msd.unsqueeze(-1).repeat_interleave(target.shape[-1], dim = -1)
    loss_items = torch.sqrt(torch.mean((target - output)**2 / (msd_reshaped + 1e-5), dim = -1))
    loss_items_clamped = torch.clamp(loss_items, 0, 5)
    loss = torch.mean(loss_items_clamped)
    
    return loss