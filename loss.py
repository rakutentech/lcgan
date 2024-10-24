import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch import autograd


def contrastive_loss(anchor, p_sample, n_sample, tau):
    b, n = anchor.size()
    exp_p_sim = torch.exp(torch.bmm(anchor.view(b, 1, n), p_sample.view(b, n, 1)).squeeze() / tau)
    exp_n_sim = torch.exp(torch.bmm(anchor.view(b, 1, n), n_sample.view(b, n, 1)).squeeze() / tau)
    loss = -torch.log(exp_p_sim / (exp_p_sim + exp_n_sim))
    loss = loss.mean()
    return loss    


def cal_r1_reg(adv_output, images, device):
    batch_size = images.size(0)
    grad_dout = cal_derivative(inputs=images, outputs=adv_output.sum(), device=device)
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == images.size())
    r1_reg = 0.5 * grad_dout2.contiguous().view(batch_size, -1).sum(1).mean(0) + images[:,0,0,0].mean()*0
    return r1_reg


def cal_derivative(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                          inputs=inputs,
                          grad_outputs=torch.ones(outputs.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    return grads
