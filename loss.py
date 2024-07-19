import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch import autograd


def d_vanilla(d_logit_real, d_logit_fake):
    d_loss = torch.mean(F.softplus(-d_logit_real)) + torch.mean(F.softplus(d_logit_fake))
    return d_loss


def g_vanilla(d_logit_fake):
    return torch.mean(F.softplus(-d_logit_fake))


def d_logistic(d_logit_real, d_logit_fake):
    d_loss = F.softplus(-d_logit_real) + F.softplus(d_logit_fake)
    return d_loss.mean()


def g_logistic(d_logit_fake):
    # basically same as g_vanilla.
    return F.softplus(-d_logit_fake).mean()


def d_contrastive_loss(anchor, p_sample, n_sample, tau, omega):
    b, n = anchor.size()
    positive = torch.exp(torch.bmm(anchor.view(b, 1, n), p_sample.view(b, n, 1)).squeeze() / tau)
    negative = torch.exp(torch.bmm(anchor.view(b, 1, n), n_sample.view(b, n, 1)).squeeze() / tau)
    loss = -torch.log(positive / (positive + negative))
    loss = loss.mean()
    return loss


def g_contrastive_loss(anchor, p_sample, n_sample, tau, omega):
    b, n = anchor.size()
    positive = torch.exp(torch.bmm(anchor.view(b, 1, n), p_sample.view(b, n, 1)).squeeze() / tau)
    negative = torch.exp(torch.bmm(anchor.view(b, 1, n), n_sample.view(b, n, 1)).squeeze() / tau)
    loss = -torch.log(positive / (positive + negative))
    loss = loss.mean()
    return loss


def sparsity_loss(x):
    loss = torch.abs(x).sum(1).mean()
    return loss


def non_diagonal_loss(x, selected_dim):
    batch_size, dims = x.size()
    mask = torch.ones_like(x).bool().cuda()
    mask[:,selected_dim] = False
    non_diagonal_elements = x.masked_select(mask).view(batch_size,dims-1)
    loss = torch.abs(non_diagonal_elements).sum(1).mean()
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

def ortho_loss(model, weight_attr='weight'):
    ortho_loss = 0.0
    total_params = 0
    
    for module in model.modules():
        if hasattr(module, weight_attr):
            W = getattr(module, weight_attr)
            if len(W.shape) == 2:
                # Ensure W is a matrix
                ortho_loss += torch.norm(torch.mm(W.t(), W) - torch.eye(W.size(1), device=W.device), 'fro')
                total_params += W.numel()

    return ortho_loss / total_params