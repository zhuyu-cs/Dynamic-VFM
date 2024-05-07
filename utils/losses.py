import torch
import numpy as np
import torch.nn.functional as F

def compute_loss(loss_type, model, model_output, responses, dataloader, data_key, detach_core=False, scale_loss=True):
    loss_scale = (
        np.sqrt(len(dataloader[data_key].dataset) / responses.shape[0])
        if scale_loss
        else 1.0
    )
    
    time_left = model_output.shape[1]

    original_data = responses.transpose(2, 1)[:, -time_left:, :]

    if loss_type=='possion':
        return PossionNLLloss(original_data, model_output,  loss_scale)
    elif loss_type=='possionv2':
        return PossionNLLlossV2(original_data, model_output,  loss_scale)
    elif loss_type=='possionv2full':
        return PossionNLLlossV2FULL(original_data, model_output,  loss_scale)

def PossionNLLloss(original_data, model_output,  loss_scale):
    
    eps=1e-8
    possion_loss = torch.sum(model_output+1e-8 - (original_data+1e-8) * torch.log(model_output+1e-8))
    assert not (torch.isnan(possion_loss).any() or torch.isinf(possion_loss).any()), "None or inf value encountered!"
    return loss_scale * possion_loss.mean()

def PossionNLLlossV2(original_data, model_output,  loss_scale):
    
    possion_loss = F.poisson_nll_loss(model_output, original_data.detach(), log_input=False, full=False, eps=1e-8, reduction="none")
    assert not (torch.isnan(possion_loss).any() or torch.isinf(possion_loss).any()), "None or inf value encountered!"
    return loss_scale * possion_loss.sum()

def PossionNLLlossV2FULL(original_data, model_output,  loss_scale):
    
    possion_loss = F.poisson_nll_loss(model_output, original_data.detach(), log_input=False, full=True, eps=1e-8, reduction="none")
    assert not (torch.isnan(possion_loss).any() or torch.isinf(possion_loss).any()), "None or inf value encountered!"
    return loss_scale * possion_loss.sum()
