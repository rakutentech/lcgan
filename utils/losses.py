# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/loss.py

from torch.nn import DataParallel
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from utils.style_ops import conv2d_gradfix
import utils.ops as ops


class GatherLayer(torch.autograd.Function):
    """
    This file is copied from
    https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    Gather tensors from all process, supporting backward propagation
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, cls_output, label, **_):
        return self.ce_loss(cls_output, label).mean()


class MiCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(MiCrossEntropyLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, mi_cls_output, label, **_):
        return self.ce_loss(mi_cls_output, label).mean()


class ConditionalContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, master_rank, DDP):
        super(ConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label, **_):
        if self.DDP:
            embed = torch.cat(GatherLayer.apply(embed), dim=0)
            proxy = torch.cat(GatherLayer.apply(proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)

        sim_matrix = self.calculate_similarity_matrix(embed, embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()


class MiConditionalContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, master_rank, DDP):
        super(MiConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, mi_embed, mi_proxy, label, **_):
        if self.DDP:
            mi_embed = torch.cat(GatherLayer.apply(mi_embed), dim=0)
            mi_proxy = torch.cat(GatherLayer.apply(mi_proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)

        sim_matrix = self.calculate_similarity_matrix(mi_embed, mi_embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(mi_embed, mi_proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()


class Data2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, m_p, master_rank, DDP):
        super(Data2DataCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.m_p = m_p
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels = labels.detach().cpu().numpy()
        num_samples = labels.shape[0]
        mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0

        for c in range(self.num_classes):
            c_indices = np.where(labels==c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def forward(self, embed, proxy, label, **_):
        # If train a GAN throuh DDP, gather all data on the master rank
        if self.DDP:
            embed = torch.cat(GatherLayer.apply(embed), dim=0)
            proxy = torch.cat(GatherLayer.apply(proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)

        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(embed, embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix/self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(embed, proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label)[label])
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn*torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy)/self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()


class MiData2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, m_p, master_rank, DDP):
        super(MiData2DataCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.m_p = m_p
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels = labels.detach().cpu().numpy()
        num_samples = labels.shape[0]
        mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0

        for c in range(self.num_classes):
            c_indices = np.where(labels==c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def forward(self, mi_embed, mi_proxy, label, **_):
        # If train a GAN throuh DDP, gather all data on the master rank
        if self.DDP:
            mi_embed = torch.cat(GatherLayer.apply(mi_embed), dim=0)
            mi_proxy = torch.cat(GatherLayer.apply(mi_proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)

        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(mi_embed, mi_embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix/self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(mi_embed, mi_proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label)[label])
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn*torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy)/self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()


class PathLengthRegularizer:
    def __init__(self, device, pl_decay=0.01, pl_weight=2):
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def cal_pl_reg(self, fake_images, ws):
        #ws refers to weight style
        #receives new fake_images of original batch (in original implementation, fakes_images used for calculating g_loss and pl_loss is generated independently)
        pl_noise = torch.randn_like(fake_images) / np.sqrt(fake_images.shape[2] * fake_images.shape[3])
        with conv2d_gradfix.no_weight_gradients():
            pl_grads = torch.autograd.grad(outputs=[(fake_images * pl_noise).sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        loss_Gpl = (pl_penalty * self.pl_weight).mean(0)
        return loss_Gpl


def enable_allreduce(dict_):
    loss = 0
    for key, value in dict_.items():
        if value is not None and key != "label":
            loss += value.mean()*0
    return loss


def d_vanilla(d_logit_real, d_logit_fake, DDP):
    d_loss = torch.mean(F.softplus(-d_logit_real)) + torch.mean(F.softplus(d_logit_fake))
    return d_loss


def g_vanilla(d_logit_fake, DDP):
    return torch.mean(F.softplus(-d_logit_fake))


def d_logistic(d_logit_real, d_logit_fake, DDP):
    d_loss = F.softplus(-d_logit_real) + F.softplus(d_logit_fake)
    return d_loss.mean()


def g_logistic(d_logit_fake, DDP):
    # basically same as g_vanilla.
    return F.softplus(-d_logit_fake).mean()


def d_ls(d_logit_real, d_logit_fake, DDP):
    d_loss = 0.5 * (d_logit_real - torch.ones_like(d_logit_real))**2 + 0.5 * (d_logit_fake)**2
    return d_loss.mean()


def g_ls(d_logit_fake, DDP):
    gen_loss = 0.5 * (d_logit_fake - torch.ones_like(d_logit_fake))**2
    return gen_loss.mean()


def d_hinge(d_logit_real, d_logit_fake, DDP):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))


def g_hinge(d_logit_fake, DDP):
    return -torch.mean(d_logit_fake)


def d_wasserstein(d_logit_real, d_logit_fake, DDP):
    return torch.mean(d_logit_fake - d_logit_real)


def g_wasserstein(d_logit_fake, DDP):
    return -torch.mean(d_logit_fake)


def crammer_singer_loss(adv_output, label, DDP, **_):
    # https://github.com/ilyakava/BigGAN-PyTorch/blob/master/train_fns.py
    # crammer singer criterion
    num_real_classes = adv_output.shape[1] - 1
    mask = torch.ones_like(adv_output).to(adv_output.device)
    mask.scatter_(1, label.unsqueeze(-1), 0)
    wrongs = torch.masked_select(adv_output, mask.bool()).reshape(adv_output.shape[0], num_real_classes)
    max_wrong, _ = wrongs.max(1)
    max_wrong = max_wrong.unsqueeze(-1)
    target = adv_output.gather(1, label.unsqueeze(-1))
    return torch.mean(F.relu(1 + max_wrong - target))


def feature_matching_loss(real_embed, fake_embed):
    # https://github.com/ilyakava/BigGAN-PyTorch/blob/master/train_fns.py
    # feature matching criterion
    fm_loss = torch.mean(torch.abs(torch.mean(fake_embed, 0) - torch.mean(real_embed, 0)))
    return fm_loss


def lecam_reg(d_logit_real, d_logit_fake, ema):
    reg = torch.mean(F.relu(d_logit_real - ema.D_fake).pow(2)) + \
          torch.mean(F.relu(ema.D_real - d_logit_fake).pow(2))
    return reg


def cal_deriv(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                          inputs=inputs,
                          grad_outputs=torch.ones(outputs.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    return grads


def latent_optimise(zs, fake_labels, generator, discriminator, batch_size, lo_rate, lo_steps, lo_alpha, lo_beta, eval,
                    cal_trsp_cost, device):
    for step in range(lo_steps - 1):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_() > 1 - lo_rate).to(device)

        zs = autograd.Variable(zs, requires_grad=True)
        fake_images = generator(zs, fake_labels, eval=eval)
        fake_dict = discriminator(fake_images, fake_labels, eval=eval)
        z_grads = cal_deriv(inputs=zs, outputs=fake_dict["adv_output"], device=device)
        z_grads_norm = torch.unsqueeze((z_grads.norm(2, dim=1)**2), dim=1)
        delta_z = lo_alpha * z_grads / (lo_beta + z_grads_norm)
        zs = torch.clamp(zs + drop_mask * delta_z, -1.0, 1.0)

        if cal_trsp_cost:
            if step == 0:
                trsf_cost = (delta_z.norm(2, dim=1)**2).mean()
            else:
                trsf_cost += (delta_z.norm(2, dim=1)**2).mean()
        else:
            trsf_cost = None
        return zs, trsf_cost


def cal_grad_penalty(real_images, real_labels, fake_images, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_images.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() + interpolates[:,0,0,0].mean()*0
    return grad_penalty


def cal_dra_penalty(real_images, real_labels, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    differences = 0.5 * real_images.std() * torch.rand(real_images.size()).to(device)
    interpolates = real_images + (alpha * differences)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() + interpolates[:,0,0,0].mean()*0
    return grad_penalty


def cal_maxgrad_penalty(real_images, real_labels, fake_images, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_images.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    maxgrad_penalty = torch.max(grads.norm(2, dim=1)**2) + interpolates[:,0,0,0].mean()*0
    return maxgrad_penalty


def cal_r1_reg(adv_output, images, device):
    batch_size = images.size(0)
    grad_dout = cal_deriv(inputs=images, outputs=adv_output.sum(), device=device)
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == images.size())
    r1_reg = 0.5 * grad_dout2.contiguous().view(batch_size, -1).sum(1).mean(0) + images[:,0,0,0].mean()*0
    return r1_reg


def adjust_k(current_k, topk_gamma, sup_k):
    current_k = max(current_k * topk_gamma, sup_k)
    return current_k


def normal_nll_loss(x, mu, var):
    # https://github.com/Natsu6767/InfoGAN-PyTorch/blob/master/utils.py
    # Calculate the negative log likelihood of normal distribution.
    # Needs to be minimized in InfoGAN. (Treats Q(c]x) as a factored Gaussian)
    logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
    nll = -(logli.sum(1).mean())
    return nll


def stylegan_cal_r1_reg(adv_output, images):
    with conv2d_gradfix.no_weight_gradients():
        r1_grads = torch.autograd.grad(outputs=[adv_output.sum()], inputs=[images], create_graph=True, only_inputs=True)[0]
    r1_penalty = r1_grads.square().sum([1,2,3]) / 2
    return r1_penalty.mean()


def d_pairwise_contrastive(d_real_logit, d_fake_logit,
                           real_content, geo_content, app_content,
                           real_style, geo_style, app_style,
                           omega, tau, aug_weight, DDP):

    adv_loss = d_logistic(d_real_logit, d_fake_logit, DDP)
    b, n = real_content.size()
    diagonal = torch.eye(b, dtype=bool).cuda()
    non_diagonal = ~torch.eye(b, dtype=bool).cuda()
    content_sim = torch.exp(torch.mm(real_content, torch.transpose(app_content, 1, 0)) / tau)
    content_pos = content_sim.masked_select(diagonal).view(b)
    content_neg = content_sim.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)
    content_neg2 = torch.exp(torch.mm(real_content, torch.transpose(geo_content, 1, 0)) / tau).sum(dim=1)

    b, n = real_style.size()
    diagonal = torch.eye(b, dtype=bool).cuda()
    non_diagonal = ~torch.eye(b, dtype=bool).cuda()
    style_sim = torch.exp(torch.mm(real_style, torch.transpose(geo_style, 1, 0)) / tau)
    style_pos = style_sim.masked_select(diagonal).view(b)
    style_neg = style_sim.masked_select(non_diagonal).view(b, b-1).sum(dim=1)
    style_neg2 = torch.exp(torch.mm(real_style, torch.transpose(app_style, 1, 0)) / tau).sum(dim=1)

    content_loss = -torch.log(content_pos / (content_pos + content_neg + content_neg2)).mean()
    style_loss = -torch.log(style_pos / (style_pos + style_neg + style_neg2)).mean()

    contrastive_loss = (content_loss + style_loss) * aug_weight
    total_d_loss = adv_loss + contrastive_loss

    return total_d_loss, content_loss, style_loss


def g_pairwise_contrastive(logit_aa, logit_ab, logit_ba, logit_bb,
                           content_aa, content_ab, content_ba, content_bb,
                           style_aa, style_ab, style_ba, style_bb,
                           omega, tau, aug_weight, DDP):

    adv_loss = g_logistic(logit_aa, DDP) + g_logistic(logit_ab, DDP) \
               + g_logistic(logit_ba, DDP) + g_logistic(logit_bb, DDP)

    b, n = content_aa.size()
    diagonal = torch.eye(b, dtype=bool).cuda()
    non_diagonal = ~torch.eye(b, dtype=bool).cuda()

    content_sim_aa = torch.exp(torch.mm(content_aa, torch.transpose(content_ab, 1, 0)) / tau)
    content_sim_ab = torch.exp(torch.mm(content_ab, torch.transpose(content_aa, 1, 0)) / tau)
    content_sim_ba = torch.exp(torch.mm(content_ba, torch.transpose(content_bb, 1, 0)) / tau)
    content_sim_bb = torch.exp(torch.mm(content_bb, torch.transpose(content_ba, 1, 0)) / tau)

    content_pos_aa = content_sim_aa.masked_select(diagonal).view(b)
    content_neg_aa = content_sim_aa.masked_select(non_diagonal).view(b, b-1).sum(dim=1)
    content_pos_ab = content_sim_ab.masked_select(diagonal).view(b)
    content_neg_ab = content_sim_ab.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)
    content_pos_ba = content_sim_ba.masked_select(diagonal).view(b)
    content_neg_ba = content_sim_ba.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)
    content_pos_bb = content_sim_bb.masked_select(diagonal).view(b)
    content_neg_bb = content_sim_bb.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)

    content_neg_aa2 = torch.exp(torch.mm(content_aa, torch.transpose(content_ba, 1, 0)) / tau).sum(dim=1)
    content_neg_ab2 = torch.exp(torch.mm(content_ab, torch.transpose(content_bb, 1, 0)) / tau).sum(dim=1)
    content_neg_ba2 = torch.exp(torch.mm(content_ba, torch.transpose(content_aa, 1, 0)) / tau).sum(dim=1)
    content_neg_bb2 = torch.exp(torch.mm(content_bb, torch.transpose(content_ab, 1, 0)) / tau).sum(dim=1)

    content_aa_loss = -torch.log(content_pos_aa / (content_pos_aa + content_neg_aa + content_neg_aa2))
    content_ab_loss = -torch.log(content_pos_ab / (content_pos_ab + content_neg_ab + content_neg_ab2))
    content_ba_loss = -torch.log(content_pos_ba / (content_pos_ba + content_neg_ba + content_neg_ba2))
    content_bb_loss = -torch.log(content_pos_bb / (content_pos_bb + content_neg_bb + content_neg_bb2))

    b, n = style_aa.size()
    diagonal = torch.eye(b, dtype=bool).cuda()
    non_diagonal = ~torch.eye(b, dtype=bool).cuda()
    style_sim_aa = torch.exp(torch.mm(style_aa, torch.transpose(style_ba, 1, 0)) / tau)
    style_sim_ab = torch.exp(torch.mm(style_ab, torch.transpose(style_bb, 1, 0)) / tau)
    style_sim_ba = torch.exp(torch.mm(style_ba, torch.transpose(style_aa, 1, 0)) / tau)
    style_sim_bb = torch.exp(torch.mm(style_bb, torch.transpose(style_ab, 1, 0)) / tau)

    style_pos_aa = style_sim_aa.masked_select(diagonal).view(b)
    style_neg_aa = style_sim_aa.masked_select(non_diagonal).view(b, b-1).sum(dim=1)
    style_pos_ab = style_sim_ab.masked_select(diagonal).view(b)
    style_neg_ab = style_sim_ab.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)
    style_pos_ba = style_sim_ba.masked_select(diagonal).view(b)
    style_neg_ba = style_sim_ba.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)
    style_pos_bb = style_sim_bb.masked_select(diagonal).view(b)
    style_neg_bb = style_sim_bb.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)

    style_neg_aa2 = torch.exp(torch.mm(style_aa, torch.transpose(style_ab, 1, 0)) / tau).sum(dim=1)
    style_neg_ab2 = torch.exp(torch.mm(style_ab, torch.transpose(style_aa, 1, 0)) / tau).sum(dim=1)
    style_neg_ba2 = torch.exp(torch.mm(style_ba, torch.transpose(style_bb, 1, 0)) / tau).sum(dim=1)
    style_neg_bb2 = torch.exp(torch.mm(style_bb, torch.transpose(style_ba, 1, 0)) / tau).sum(dim=1)

    style_aa_loss = -torch.log(style_pos_aa / (style_pos_aa + style_neg_aa + style_neg_aa2))
    style_ab_loss = -torch.log(style_pos_ab / (style_pos_ab + style_neg_ab + style_neg_ab2))
    style_ba_loss = -torch.log(style_pos_ba / (style_pos_ba + style_neg_ba + style_neg_ba2))
    style_bb_loss = -torch.log(style_pos_bb / (style_pos_bb + style_neg_bb + style_neg_bb2))

    content_loss = content_aa_loss.mean() + content_ab_loss.mean() + content_ba_loss.mean() + content_bb_loss.mean()
    style_loss = style_aa_loss.mean() + style_ab_loss.mean() + style_ba_loss.mean() + style_bb_loss.mean()

    contrastive_loss = (content_loss + style_loss) * aug_weight
    total_gen_loss = (adv_loss + contrastive_loss) / 4.0

    return total_gen_loss, content_loss, style_loss


def d_triplet_contrastive(d_real_logit, d_fake_logit,
                          real_content, geo_content, app_content,
                          real_style, geo_style, app_style,
                          omega, tau, aug_weight, DDP):

    adv_loss = d_logistic(d_real_logit, d_fake_logit, DDP)
    b, n = real_content.size()
    diagonal = torch.eye(b, dtype=bool).cuda()
    non_diagonal = ~torch.eye(b, dtype=bool).cuda()
    content_pos = torch.exp(torch.bmm(real_content.view(b, 1, n), app_content.view(b, n, 1)).squeeze() / tau)
    content_neg = torch.exp(torch.mm(real_content, torch.transpose(geo_content, 1, 0)) / tau)
    content_strong_neg = content_neg.masked_select(diagonal).view(b)
    content_weak_neg = content_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)*omega

    b, n = real_style.size()
    diagonal = torch.eye(b, dtype=bool).cuda()
    non_diagonal = ~torch.eye(b, dtype=bool).cuda()
    style_pos = torch.exp(torch.bmm(real_style.view(b, 1, n), geo_style.view(b, n, 1)).squeeze() / tau)
    style_neg = torch.exp(torch.mm(real_style, torch.transpose(app_style, 1, 0)) / tau)
    style_strong_neg = style_neg.masked_select(diagonal).view(b)
    style_weak_neg = style_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1)*omega

    content_loss = -torch.log(content_pos / (content_pos + content_strong_neg + content_weak_neg)).mean()
    style_loss = -torch.log(style_pos / (style_pos + style_strong_neg + style_weak_neg)).mean()

    contrastive_loss = (content_loss + style_loss) * aug_weight
    total_d_loss = adv_loss + contrastive_loss

    return total_d_loss, content_loss, style_loss


def g_triplet_contrastive(logit_ac, logit_ad, logit_bc, logit_bd,
                          content_aa, content_ab, content_ba, content_bb,
                          style_aa, style_ab, style_ba, style_bb,
                          omega, tau, aug_weight, DDP):

    adv_loss = g_logistic(logit_ac, DDP) + g_logistic(logit_ad, DDP) \
               + g_logistic(logit_bc, DDP) + g_logistic(logit_bd, DDP)

    b, n = content_aa.size()
    diagonal = torch.eye(b, dtype=bool).cuda()
    non_diagonal = ~torch.eye(b, dtype=bool).cuda()

    content_aa_pos = torch.exp(torch.bmm(content_aa.view(b, 1, n), content_ab.view(b, n, 1)).squeeze() / tau)
    content_aa_neg = torch.exp(torch.mm(content_aa, torch.transpose(content_ba, 1, 0)) / tau)
    content_aa_strong_neg = content_aa_neg.masked_select(diagonal).view(b)
    content_aa_weak_neg = content_aa_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1) * omega

    content_ab_pos = torch.exp(torch.bmm(content_ab.view(b, 1, n), content_aa.view(b, n, 1)).squeeze() / tau)
    content_ab_neg = torch.exp(torch.mm(content_ab, torch.transpose(content_bb, 1, 0)) / tau)
    content_ab_strong_neg = content_ab_neg.masked_select(diagonal).view(b)
    content_ab_weak_neg = content_ab_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1) * omega

    content_ba_pos = torch.exp(torch.bmm(content_ba.view(b, 1, n), content_bb.view(b, n, 1)).squeeze() / tau)
    content_ba_neg = torch.exp(torch.mm(content_ba, torch.transpose(content_aa, 1, 0)) / tau)
    content_ba_strong_neg = content_ba_neg.masked_select(diagonal).view(b)
    content_ba_weak_neg = content_ba_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1) * omega

    content_bb_pos = torch.exp(torch.bmm(content_bb.view(b, 1, n), content_ba.view(b, n, 1)).squeeze() / tau)
    content_bb_neg = torch.exp(torch.mm(content_bb, torch.transpose(content_ab, 1, 0)) / tau)
    content_bb_strong_neg = content_bb_neg.masked_select(diagonal).view(b)
    content_bb_weak_neg = content_bb_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1) * omega

    content_aa_loss = -torch.log(content_aa_pos / (content_aa_pos + content_aa_strong_neg + content_aa_weak_neg))
    content_ab_loss = -torch.log(content_ab_pos / (content_ab_pos + content_ab_strong_neg + content_ab_weak_neg))
    content_ba_loss = -torch.log(content_ba_pos / (content_ba_pos + content_ba_strong_neg + content_ba_weak_neg))
    content_bb_loss = -torch.log(content_bb_pos / (content_bb_pos + content_bb_strong_neg + content_bb_weak_neg))

    b, n = style_aa.size()
    style_aa_pos = torch.exp(torch.bmm(style_aa.view(b, 1, n), style_ba.view(b, n, 1)).squeeze() / tau)
    style_aa_neg = torch.exp(torch.mm(style_aa, torch.transpose(style_ab, 1, 0)) / tau)
    style_aa_strong_neg = style_aa_neg.masked_select(diagonal).view(b)
    style_aa_weak_neg = style_aa_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1) * omega

    style_ab_pos = torch.exp(torch.bmm(style_ab.view(b, 1, n), style_bb.view(b, n, 1)).squeeze() / tau)
    style_ab_neg = torch.exp(torch.mm(style_ab, torch.transpose(style_aa, 1, 0)) / tau)
    style_ab_strong_neg = style_ab_neg.masked_select(diagonal).view(b)
    style_ab_weak_neg = style_ab_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1) * omega

    style_ba_pos = torch.exp(torch.bmm(style_ba.view(b, 1, n), style_aa.view(b, n, 1)).squeeze() / tau)
    style_ba_neg = torch.exp(torch.mm(style_ba, torch.transpose(style_bb, 1, 0)) / tau)
    style_ba_strong_neg = style_ba_neg.masked_select(diagonal).view(b)
    style_ba_weak_neg = style_ba_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1) * omega

    style_bb_pos = torch.exp(torch.bmm(style_bb.view(b, 1, n), style_ab.view(b, n, 1)).squeeze() / tau)
    style_bb_neg = torch.exp(torch.mm(style_bb, torch.transpose(style_ba, 1, 0)) / tau)
    style_bb_strong_neg = style_bb_neg.masked_select(diagonal).view(b)
    style_bb_weak_neg = style_bb_neg.masked_select(non_diagonal).view(b, b - 1).sum(dim=1) * omega

    style_aa_loss = -torch.log(style_aa_pos / (style_aa_pos + style_aa_strong_neg + style_aa_weak_neg))
    style_ab_loss = -torch.log(style_ab_pos / (style_ab_pos + style_ab_strong_neg + style_ab_weak_neg))
    style_ba_loss = -torch.log(style_ba_pos / (style_ba_pos + style_ba_strong_neg + style_ba_weak_neg))
    style_bb_loss = -torch.log(style_bb_pos / (style_bb_pos + style_bb_strong_neg + style_bb_weak_neg))

    content_loss = (content_aa_loss.mean() + content_ab_loss.mean() +
                    content_ba_loss.mean() + content_bb_loss.mean()) * 0.25

    style_loss = (style_aa_loss.mean() + style_ab_loss.mean() +
                  style_ba_loss.mean() + style_bb_loss.mean()) * 0.25

    contrastive_loss = (content_loss + style_loss) * aug_weight
    total_gen_loss = adv_loss/4.0 + contrastive_loss

    return total_gen_loss, content_loss, style_loss

# def d_triplet_contrastive(d_real_logit, d_fake_logit,
#                           real_content, geo_content, app_content,
#                           real_style, geo_style, app_style,
#                           omega, tau, aug_weight, DDP):
#
#     adv_loss = d_logistic(d_real_logit, d_fake_logit, DDP)
#     b, n = real_content.size()
#     content_positive = torch.exp(torch.bmm(real_content.view(b, 1, n), app_content.view(b, n, 1)).squeeze() / tau)
#     content_negative = torch.exp(torch.bmm(real_content.view(b, 1, n), geo_content.view(b, n, 1)).squeeze() / tau)
#
#     b, n = real_style.size()
#     style_positive = torch.exp(torch.bmm(real_style.view(b, 1, n), geo_style.view(b, n, 1)).squeeze() / tau)
#     style_negative = torch.exp(torch.bmm(real_style.view(b, 1, n), app_style.view(b, n, 1)).squeeze() / tau)
#
#     content_loss = -torch.log(content_positive / (content_positive + content_negative)).mean()
#     style_loss = -torch.log(style_positive / (style_positive + style_negative)).mean()
#
#     contrastive_loss = (content_loss + style_loss) * aug_weight
#     total_d_loss = adv_loss + contrastive_loss
#
#     return total_d_loss, content_loss, style_loss
#
#
# def g_triplet_contrastive(logit_ac, logit_ad, logit_bc, logit_bd,
#                           content_aa, content_ab, content_ba, content_bb,
#                           style_aa, style_ab, style_ba, style_bb,
#                           omega, tau, aug_weight, DDP):
#
#     adv_loss = g_logistic(logit_ac, DDP) + g_logistic(logit_ad, DDP) \
#                + g_logistic(logit_bc, DDP) + g_logistic(logit_bd, DDP)
#
#     b, n = content_aa.size()
#     content_aa_positive = torch.exp(torch.bmm(content_aa.view(b, 1, n), content_ab.view(b, n, 1)).squeeze() / tau)
#     content_aa_negative = torch.exp(torch.bmm(content_aa.view(b, 1, n), content_ba.view(b, n, 1)).squeeze() / tau)
#
#     content_ab_positive = torch.exp(torch.bmm(content_ab.view(b, 1, n), content_aa.view(b, n, 1)).squeeze() / tau)
#     content_ab_negative = torch.exp(torch.bmm(content_ab.view(b, 1, n), content_bb.view(b, n, 1)).squeeze() / tau)
#
#     content_ba_positive = torch.exp(torch.bmm(content_ba.view(b, 1, n), content_bb.view(b, n, 1)).squeeze() / tau)
#     content_ba_negative = torch.exp(torch.bmm(content_ba.view(b, 1, n), content_aa.view(b, n, 1)).squeeze() / tau)
#
#     content_bb_positive = torch.exp(torch.bmm(content_bb.view(b, 1, n), content_ba.view(b, n, 1)).squeeze() / tau)
#     content_bb_negative = torch.exp(torch.bmm(content_bb.view(b, 1, n), content_ab.view(b, n, 1)).squeeze() / tau)
#
#     content_aa_loss = -torch.log(content_aa_positive / (content_aa_positive + content_aa_negative))
#     content_ab_loss = -torch.log(content_ab_positive / (content_ab_positive + content_ab_negative))
#     content_ba_loss = -torch.log(content_ba_positive / (content_ba_positive + content_ba_negative))
#     content_bb_loss = -torch.log(content_bb_positive / (content_bb_positive + content_bb_negative))
#
#     b, n = style_aa.size()
#     style_aa_positive = torch.exp(torch.bmm(style_aa.view(b, 1, n), style_ba.view(b, n, 1)).squeeze() / tau)
#     style_aa_negative = torch.exp(torch.bmm(style_aa.view(b, 1, n), style_ab.view(b, n, 1)).squeeze() / tau)
#
#     style_ab_positive = torch.exp(torch.bmm(style_ab.view(b, 1, n), style_bb.view(b, n, 1)).squeeze() / tau)
#     style_ab_negative = torch.exp(torch.bmm(style_ab.view(b, 1, n), style_aa.view(b, n, 1)).squeeze() / tau)
#
#     style_ba_positive = torch.exp(torch.bmm(style_ba.view(b, 1, n), style_aa.view(b, n, 1)).squeeze() / tau)
#     style_ba_negative = torch.exp(torch.bmm(style_ba.view(b, 1, n), style_bb.view(b, n, 1)).squeeze() / tau)
#
#     style_bb_positive = torch.exp(torch.bmm(style_bb.view(b, 1, n), style_ab.view(b, n, 1)).squeeze() / tau)
#     style_bb_negative = torch.exp(torch.bmm(style_bb.view(b, 1, n), style_ba.view(b, n, 1)).squeeze() / tau)
#
#     style_aa_loss = -torch.log(style_aa_positive / (style_aa_positive + style_aa_negative))
#     style_ab_loss = -torch.log(style_ab_positive / (style_ab_positive + style_ab_negative))
#     style_ba_loss = -torch.log(style_ba_positive / (style_ba_positive + style_ba_negative))
#     style_bb_loss = -torch.log(style_bb_positive / (style_bb_positive + style_bb_negative))
#
#     content_loss = (content_aa_loss.mean() + content_ab_loss.mean() +
#                     content_ba_loss.mean() + content_bb_loss.mean()) * 0.25
#
#     style_loss = (style_aa_loss.mean() + style_ab_loss.mean() +
#                   style_ba_loss.mean() + style_bb_loss.mean()) * 0.25
#
#     contrastive_loss = (content_loss + style_loss) * aug_weight
#     total_gen_loss = adv_loss/4.0 + contrastive_loss
#
#     return total_gen_loss, content_loss, style_loss
