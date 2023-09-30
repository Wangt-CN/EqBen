import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import random

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils import all_gather
import torch.distributed as dist


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack([ti if itm_labels[i] == 1 else fi for i, (ti, fi) in enumerate(zip(bti, bfi))])
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


class pairloss_criterion(nn.Module):
    def __init__(self, text_side=False, img_side=False, group_side=False, pos_reg=False, norm=False, weight=1.0, margin=0.0):
        super(pairloss_criterion, self).__init__()
        self.text_side = text_side
        self.img_side = img_side
        self.group_side = group_side
        self.pos_reg = pos_reg
        self.norm = norm
        self.weight = weight
        self.margin = margin # smaller than margin -> loss=0.
        self.mseloss = nn.MSELoss()
        self.relu = nn.ReLU(inplace=False)

    def __call__(self, c0_i0, c1_i0, c0_i1, c1_i1):
        loss = 0.
        if self.text_side:
            if self.norm: # norm just use weight=1.0
                loss += (self.mseloss((c0_i0-c1_i0)/(c0_i0.abs()), (c1_i1-c0_i1)/(c1_i1.abs())) - self.margin).clamp(min=0.)
            else:
                loss += self.weight * (self.mseloss((c0_i0-c1_i0), (c1_i1-c0_i1)) - self.margin).clamp(min=0.)
            if self.pos_reg:
                loss += self.relu(-(c0_i0-c1_i0)).mean() + self.relu(-(c1_i1-c0_i1)).mean()

        if self.img_side:
            if self.norm: # norm just use weight=1.0
                loss += (self.mseloss((c0_i0-c0_i1)/(c0_i0.abs()), (c1_i1-c1_i0)/(c1_i1.abs())) - self.margin).clamp(min=0.)
            else:
                loss += self.weight * (self.mseloss((c0_i0-c0_i1), (c1_i1-c1_i0)) - self.margin).clamp(min=0.)
            if self.pos_reg:
                loss += self.relu(-(c0_i0-c0_i1)).mean() + self.relu(-(c1_i1-c1_i0)).mean()

        if self.group_side:
            if self.norm: # norm just use weight=1.0
                loss += (self.mseloss((c0_i0-c0_i1)/(c0_i0.abs()), (c0_i0-c1_i0)/(c0_i0.abs())) - self.margin).clamp(min=0.) + (self.mseloss((c1_i1-c0_i1)/(c1_i1.abs()), (c1_i1-c1_i0)/(c1_i1.abs())) - self.margin).clamp(min=0.)
            else:
                loss += self.weight * ((self.mseloss((c0_i0-c0_i1), (c0_i0-c1_i0)) - self.margin).clamp(min=0.) + (self.mseloss((c1_i1-c0_i1), (c1_i1-c1_i0)) - self.margin).clamp(min=0.))
            if self.pos_reg:
                loss += self.relu(-(c0_i0-c0_i1)).mean() + self.relu(-(c0_i0-c1_i0)).mean() + self.relu(-(c1_i1-c1_i0)).mean() + self.relu(-(c1_i1-c0_i1)).mean()

        return loss



class pairloss_criterion_v2(nn.Module): # margin before mean
    def __init__(self, text_side=False, img_side=False, group_side=False, pos_reg=False, norm=False, weight=1.0, margin=0.0):
        super(pairloss_criterion_v2, self).__init__()
        self.text_side = text_side
        self.img_side = img_side
        self.group_side = group_side
        self.pos_reg = pos_reg
        self.norm = norm
        self.weight = weight
        self.margin = margin # smaller than margin -> loss=0.
        self.mseloss = nn.MSELoss(reduction='none')
        self.relu = nn.ReLU(inplace=False)

    def __call__(self, c0_i0, c1_i0, c0_i1, c1_i1):
        loss = 0.
        if self.text_side:
            if self.norm: # norm just use weight=1.0
                loss += (self.mseloss((c0_i0-c1_i0)/(c0_i0.abs()), (c1_i1-c0_i1)/(c1_i1.abs())) - self.margin).clamp(min=0.)
            else:
                loss += self.weight * (self.mseloss((c0_i0-c1_i0), (c1_i1-c0_i1)) - self.margin).clamp(min=0.).mean()
            if self.pos_reg:
                loss += self.relu(-(c0_i0-c1_i0)).mean() + self.relu(-(c1_i1-c0_i1)).mean()

        if self.img_side:
            if self.norm: # norm just use weight=1.0
                loss += (self.mseloss((c0_i0-c0_i1)/(c0_i0.abs()), (c1_i1-c1_i0)/(c1_i1.abs())) - self.margin).clamp(min=0.)
            else:
                loss += self.weight * (self.mseloss((c0_i0-c0_i1), (c1_i1-c1_i0)) - self.margin).clamp(min=0.).mean()
            if self.pos_reg:
                loss += self.relu(-(c0_i0-c0_i1)).mean() + self.relu(-(c1_i1-c1_i0)).mean()

        if self.group_side:
            if self.norm: # norm just use weight=1.0
                loss += (self.mseloss((c0_i0-c0_i1)/(c0_i0.abs()), (c0_i0-c1_i0)/(c0_i0.abs())) - self.margin).clamp(min=0.) + (self.mseloss((c1_i1-c0_i1)/(c1_i1.abs()), (c1_i1-c1_i0)/(c1_i1.abs())) - self.margin).clamp(min=0.)
            else:
                loss += self.weight * ((self.mseloss((c0_i0-c0_i1), (c0_i0-c1_i0)) - self.margin).clamp(min=0.) + (self.mseloss((c1_i1-c0_i1), (c1_i1-c1_i0)) - self.margin).clamp(min=0.))
            if self.pos_reg:
                loss += self.relu(-(c0_i0-c0_i1)).mean() + self.relu(-(c0_i0-c1_i0)).mean() + self.relu(-(c1_i1-c1_i0)).mean() + self.relu(-(c1_i1-c0_i1)).mean()

        return loss


def get_pairloss_itc(loss_criterion, score_matrix):
    score_matrix_size = score_matrix.size(0) # score matrix i2t
    caption0_image0_score, caption0_imagei_score, captioni_image0_score, captioni_imagei_score = [], [], [], []
    for idx in range(score_matrix_size - 1):
        caption0_image0_score.append(score_matrix[idx, idx].unsqueeze(0).expand(score_matrix_size-(idx+1)))
        caption0_imagei_score.append(score_matrix[idx+1:, idx])
        captioni_image0_score.append(score_matrix[idx, idx+1:])
        captioni_imagei_score.append(score_matrix.diag()[idx+1:])

    caption0_image0_score = torch.cat(caption0_image0_score, dim=-1).unsqueeze(1)
    caption0_imagei_score = torch.cat(caption0_imagei_score, dim=-1).unsqueeze(1)
    captioni_image0_score = torch.cat(captioni_image0_score, dim=-1).unsqueeze(1)
    captioni_imagei_score = torch.cat(captioni_imagei_score, dim=-1).unsqueeze(1)

    # use full batch
    ### matrix-based
    pairloss = loss_criterion(caption0_image0_score, captioni_image0_score, caption0_imagei_score, captioni_imagei_score)

    return pairloss


def get_pairloss_itc_hard(loss_criterion, score_matrix, t2i_neg_idx, i2t_neg_idx, bsz, sample_num, return_mask=False):
    score_matrix_size = bsz # score matrix i2t
    global_batch_size = score_matrix.size(1)
    mask = torch.ones(global_batch_size, global_batch_size).to(score_matrix.device)

    caption0_image0_score, caption0_imagei_score, captioni_image0_score, captioni_imagei_score = [], [], [], []
    for idx in range(score_matrix_size):
        if sample_num>0:
            idx_combine = torch.cat([t2i_neg_idx[idx], i2t_neg_idx[idx]])
        else:
            idx_combine = [t2i_neg_idx[idx], i2t_neg_idx[idx]]
        caption0_image0_score.append(score_matrix[idx, idx].unsqueeze(0).expand(len(idx_combine))) # 2 denotes both image/text-side
        caption0_imagei_score.append(score_matrix[idx_combine, idx])
        captioni_image0_score.append(score_matrix[idx, idx_combine])
        captioni_imagei_score.append(score_matrix.diag()[idx_combine])

        if return_mask:
            mask[idx, idx_combine] = 0.
            mask[idx_combine, idx] = 0.

    caption0_image0_score = torch.cat(caption0_image0_score, dim=-1).unsqueeze(1)
    caption0_imagei_score = torch.cat(caption0_imagei_score, dim=-1).unsqueeze(1)
    captioni_image0_score = torch.cat(captioni_image0_score, dim=-1).unsqueeze(1)
    captioni_imagei_score = torch.cat(captioni_imagei_score, dim=-1).unsqueeze(1)

    # use full batch
    ### matrix-based
    pairloss = loss_criterion(caption0_image0_score, captioni_image0_score, caption0_imagei_score, captioni_imagei_score)

    if return_mask:
        return pairloss, mask
    else:
        return pairloss


def get_pairloss_v1_itc_hard(score_i2t, easy_mask, pl_module):
    pair_loss_v1_orig = nn.MSELoss()(score_i2t * easy_mask, score_i2t.t() * easy_mask)
    pair_loss_v1 = (pl_module.hparams.config["pairloss_weight"] / pl_module.hparams.config["pairloss_prop"]) * (pair_loss_v1_orig - pl_module.hparams.config["pairloss_margin_v1"]).clamp(min=0.)
    return pair_loss_v1



def get_pairloss_itm(loss_criterion, score, false_text_len, false_img_len, sample_num=3, activation=None):
    assert score.size(1) == 1 + false_text_len + 2*false_img_len
    if activation == 'linear': # normalize score scale to 0,1
        score_min, score_max = score.min(), score.max()
        score = (score-score_min)/(score_max-score_min)
    elif activation == 'sigmoid':
        score = F.sigmoid(score)
    elif activation == 'softmax':
        assert sample_num > 0 # softmax can only for non-batch-wise operation

    caption0_image0_score = score[:,0]
    caption0_imagei_score = score[:,1:1+false_img_len]
    captioni_image0_score = score[:,1+false_img_len:1+false_text_len+false_img_len]
    captioni_imagei_score = score[:,1+false_text_len+false_img_len:]
    score_size = false_img_len


    if sample_num > 0:
        ### random choose [sample num] to get the loss
        pairloss = 0
        sample_idx_list = random.sample(list(range(false_img_len)), sample_num)
        for sample_idx in sample_idx_list:
            c0_i0, ci_i0, c0_ii, ci_ii = caption0_image0_score, captioni_image0_score[:,sample_idx], caption0_imagei_score[:,sample_idx], captioni_imagei_score[:,sample_idx]
            if activation == 'softmax':
                c0_i0, ci_i0, c0_ii, ci_ii = torch.split(F.softmax(torch.stack([c0_i0, ci_i0, c0_ii, ci_ii], dim=-1), dim=-1), [1,1,1,1], dim=-1)
            pairloss += loss_criterion(c0_i0, ci_i0, c0_ii, ci_ii)
        pairloss /= sample_num

    elif sample_num == -1: # use full batch
        ### matrix-based
        bs = len(caption0_image0_score)
        caption0_image0_score = caption0_image0_score.unsqueeze(1).expand(bs, score_size)
        pairloss = loss_criterion(caption0_image0_score, captioni_image0_score, caption0_imagei_score, captioni_imagei_score)

    else:
        raise NotImplementedError

    return pairloss


def compute_itm_hardneg(pl_module, batch, sim_i2t, sim_t2i, logit_scale_exp, pairloss=None):
    is_training_phase = pl_module.training
    pos_len = batch["text_ids_mlm"].size(0)
    neg_len = batch["text_ids_mlm"].size(0)
    bsz = batch["text_ids_mlm"].size(0)
    sample_num = pl_module.hparams.config["sample_num"]
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len), torch.zeros(neg_len)]).to(pl_module.device)

    batch = {k: v for k, v in batch.items()}
    infer_pos = pl_module.infer(batch, mask_text=False, mask_image=False)

    batch_text_ids = infer_pos["text_ids"]
    batch_text_masks = infer_pos["text_masks"]
    batch_image = infer_pos["image"]

    with torch.no_grad():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more hard negative candidates.
        gathered_text_ids = [torch.zeros_like(batch_text_ids) for _ in range(world_size)]
        gathered_text_masks = [torch.zeros_like(batch_text_masks) for _ in range(world_size)]
        gathered_image = [torch.zeros_like(batch_image) for _ in range(world_size)]

        dist.all_gather(gathered_text_ids, batch_text_ids)
        dist.all_gather(gathered_text_masks, batch_text_masks)
        dist.all_gather(gathered_image, batch_image)

        all_text_ids = torch.cat([batch_text_ids] + gathered_text_ids[:rank] + gathered_text_ids[rank + 1 :])
        all_text_masks = torch.cat([batch_text_masks] + gathered_text_masks[:rank] + gathered_text_masks[rank + 1 :])
        all_image = torch.cat([batch_image] + gathered_image[:rank] + gathered_image[rank + 1 :])

    if pairloss == 'itc' and is_training_phase:
        score_i2t = sim_i2t[:bsz, :]
        if pl_module.hparams.config["score_pre_softmax"]:
            score_i2t = F.softmax(score_i2t, dim=1)
        pair_loss = get_pairloss_itc(pl_module.loss_func, score_i2t[:bsz, :bsz])
    elif pairloss in ['cross_consist', 'incross_consist'] and is_training_phase:
        pair_loss = pl_module.hparams.config["pairloss_weight"] * (sim_i2t - sim_t2i).square().mean() / (logit_scale_exp * logit_scale_exp) * len(sim_i2t)


    with torch.no_grad():
        weights_i2t = F.softmax(sim_i2t[:bsz, :], dim=1)
        weights_t2i = F.softmax(sim_t2i[:bsz, :], dim=1)

        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)

    # select a negative text for each image / select a negative image for each text
    images_neg, text_ids_neg, text_masks_neg = [], [], []
    images_neg_idx, text_ids_neg_idx = [], []
    for b in range(bsz):
        if sample_num>0:
            neg_idx_topk = torch.multinomial(weights_t2i[b], sample_num)
            neg_idx = neg_idx_topk[0].item()
        else:
            neg_idx_topk = neg_idx = torch.multinomial(weights_t2i[b], 1).item()

        images_neg_idx.append(neg_idx_topk)
        images_neg.append(all_image[neg_idx])
        if pairloss == 'hard_itm' and is_training_phase: # should use same negatve img/txt (txt follow the img)
            text_ids_neg.append(all_text_ids[neg_idx])
            text_masks_neg.append(all_text_masks[neg_idx])
        else:
            if sample_num > 0:
                neg_idx_topk = torch.multinomial(weights_i2t[b], sample_num)
                neg_idx = neg_idx_topk[0].item()
            else:
                neg_idx_topk = neg_idx = torch.multinomial(weights_i2t[b], 1).item()

            text_ids_neg_idx.append(neg_idx_topk)
            text_ids_neg.append(all_text_ids[neg_idx])
            text_masks_neg.append(all_text_masks[neg_idx])

    if pairloss == 'hard_easy_itc' and is_training_phase:
        score_i2t = sim_i2t # [:bsz, :] --- not to choose the own part
        if pl_module.hparams.config["score_pre_softmax"]:
            score_i2t = F.softmax(score_i2t, dim=1)
        pair_loss_v2, easy_mask = get_pairloss_itc_hard(pl_module.loss_func, score_i2t, images_neg_idx, text_ids_neg_idx, bsz=bsz, sample_num=sample_num, return_mask=True)
        pair_loss_v1 = get_pairloss_v1_itc_hard(score_i2t, easy_mask, pl_module)
        pair_loss = pair_loss_v2 + pair_loss_v1

    images_neg = torch.stack(images_neg, dim=0)
    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_masks_neg = torch.stack(text_masks_neg, dim=0)

    batch_imgs_neg = {
        "image": [images_neg],
        "text_ids": batch["text_ids"],
        "text_labels": batch["text_labels"],
        "text_masks": batch["text_masks"],
    }
    infer_imags_neg = pl_module.infer(batch_imgs_neg, mask_text=False, mask_image=False)

    batch_text_neg = {
        "image": batch["image"],
        "text_ids": text_ids_neg,
        "text_labels": batch["text_labels"],
        "text_masks": text_masks_neg,
    }
    infer_text_neg = pl_module.infer(batch_text_neg, mask_text=False, mask_image=False)


    if pairloss == 'hard_itm' and is_training_phase: # add the [neg/neg] itm forward
        batch_imgstxt_neg = {
            "image": [images_neg],
            "text_ids": text_ids_neg,
            "text_labels": batch["text_labels"],
            "text_masks": text_masks_neg,
        }
        infer_imagstxt_neg = pl_module.infer(batch_imgstxt_neg, mask_text=False, mask_image=False)

        all_cls_feats = torch.cat(
            [infer_pos["cls_feats"], infer_imags_neg["cls_feats"], infer_text_neg["cls_feats"], infer_imagstxt_neg["cls_feats"]], dim=0
        )
        # itm_labels = torch.cat([itm_labels, torch.ones(pos_len).to(pl_module.device)])
        itm_logits_pairloss = pl_module.itm_score(all_cls_feats)
        itm_logits = itm_logits_pairloss[:3*bsz]
    else:
        all_cls_feats = torch.cat(
            [infer_pos["cls_feats"], infer_imags_neg["cls_feats"], infer_text_neg["cls_feats"]], dim=0
        )
        itm_logits = pl_module.itm_score(all_cls_feats)

    # calculate original itm loss
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    if pairloss is not None:
        if pairloss == 'hard_itm' and is_training_phase:
            # first to transfer the logits (4*bs, 2)
            if pl_module.hparams.config["score_pre_softmax"]:
                itm_logits_pairloss_ = F.softmax(itm_logits_pairloss, dim=-1)
            else:
                itm_logits_pairloss_ = itm_logits_pairloss
            itm_logits_pair = itm_logits_pairloss_[:, 1].view(4, bsz).t() # bs * 4 * 1
            pair_loss = get_pairloss_itm(pl_module.loss_func, itm_logits_pair, false_text_len=1, false_img_len=1, sample_num=pl_module.hparams.config["sample_num"], activation=pl_module.hparams.config["score_activation"])
        elif 'itc' in pairloss and is_training_phase:
            pass
        elif 'consist' in pairloss and is_training_phase:
            pass
        else:
            pair_loss = torch.Tensor([0.]).to(itm_loss.device)
    else:
        pair_loss = torch.Tensor([0.]).to(itm_loss.device)

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
        "pair_loss": pair_loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/itm_loss", loss)
    pl_module.log(f"itm/{phase}/pair_loss", pair_loss, prog_bar=True if phase == "train" else False)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_itc(pl_module, batch):
    infer_imag = pl_module.infer(batch, mask_image=False, mask_text=False, image_only=True)
    infer_text = pl_module.infer(batch, mask_image=False, mask_text=False, text_only=True)

    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    logit_scale = pl_module.logit_scale.exp().mean()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # We gather tensors from all gpus to get more negatives to contrast with.
    gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
    gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
    dist.all_gather(gathered_image_features, image_features)
    dist.all_gather(gathered_text_features, text_features)

    all_image_features = torch.cat(
        [image_features] + gathered_image_features[:rank] + gathered_image_features[rank + 1 :]
    )
    all_text_features = torch.cat([text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1 :])

    # this is needed to send gradients back everywhere.
    logits_per_image = logit_scale * all_image_features @ all_text_features.t()
    logits_per_text = logits_per_image.t()

    ground_truth = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())

    itc_loss = (
        F.cross_entropy(logits_per_image.float(), ground_truth) + F.cross_entropy(logits_per_text.float(), ground_truth)
    ) / 2

    ret = {
        "itc_loss": itc_loss,
        "itc_i2t_logits": logits_per_image,
        "itc_t2i_logits": logits_per_text,
        "itc_labels": ground_truth,
        "itc_logit_scale": logit_scale,
    }

    if pl_module.hparams.config["pairloss"] == 'incross_consist' and pl_module.training: # in-modal consistency loss
        logits_i2i = logit_scale * all_image_features @ all_image_features.t()
        logits_t2t = logit_scale * all_text_features @ all_text_features.t()
        inmodal_pairloss = pl_module.hparams.config["pairloss_weight"] * (logits_i2i - logits_t2t).square().mean() / (logit_scale * logit_scale) * len(logits_i2i)
        ret.update({"inmodal_pair_loss": inmodal_pairloss})
        pl_module.log(f"itc/train/inmodal_pair_loss", inmodal_pairloss)

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itc_loss")(ret["itc_loss"])
    scale = getattr(pl_module, f"{phase}_itc_logit_scale")(ret["itc_logit_scale"])
    i2t_acc = getattr(pl_module, f"{phase}_itc_i2t_accuracy")(ret["itc_i2t_logits"], ret["itc_labels"])
    t2i_acc = getattr(pl_module, f"{phase}_itc_t2i_accuracy")(ret["itc_t2i_logits"], ret["itc_labels"])
    pl_module.log(f"itc/{phase}/loss", loss)
    pl_module.log(f"itc/{phase}/logit_scale", scale)
    pl_module.log(f"itc/{phase}/i2t_accuracy", i2t_acc)
    pl_module.log(f"itc/{phase}/t2i_accuracy", t2i_acc)

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(ret["vqa_logits"], ret["vqa_targets"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
    infer2 = pl_module.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(ret["nlvr2_logits"], ret["nlvr2_labels"])
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches])
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches])
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


@torch.no_grad()
def compute_itc_recall(pl_module):
    # TODO: multi-gpu support
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    iids = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append(_b['image'])
        iids += _b['img_index']
    iids = torch.tensor(iids)

    image_feats = list()
    text_feats = list()

    for img_batch in tqdm.tqdm(image_preload, desc="image feat loop"):
        with torch.cuda.amp.autocast():
            image_feat = pl_module.infer(
                {"image": [img_batch[0].to(pl_module.device)]}, 
                image_only=True)['cls_feats']
            image_feats.append(image_feat)
    image_feats = torch.cat(image_feats)    

    for txt_batch in tqdm.tqdm(text_preload, desc="text feat loop"):
        with torch.cuda.amp.autocast():
            text_feat = pl_module.infer(
                    {
                        "text_ids": txt_batch["text_ids"],
                        "text_masks": txt_batch["text_masks"],
                        "text_labels": txt_batch["text_labels"],
                    },
                    text_only=True
                )["cls_feats"]
            text_feats.append(text_feat)
    text_feats = torch.cat(text_feats)    

    
    torch.distributed.barrier()
    device = pl_module.device

    def gather_feats(y, device):
        gather_y = all_gather(y)
        gather_y = [x.to(device) for x in gather_y]
        return torch.cat(gather_y)

    tiids, tiids_i = tiids.sort()
    tiids = tiids.to(device)
    text_feats = text_feats[tiids_i]

    image_feats = gather_feats(image_feats, device)
    
    iids = gather_feats(iids, device)

    scores = image_feats @ text_feats.t()

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


@torch.no_grad()
def compute_itm_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    # import pdb; pdb.set_trace()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    # TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # import pdb; pdb.set_trace()
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b["image"][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch["text_ids"].device)

            with torch.cuda.amp.autocast():
                # score = pl_module.rank_output(
                score = pl_module.itm_score(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        img=im,
                    )["cls_feats"]
                )[:, 1]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name, log_dir, exp_name):
    print('start saving vqa results to {}'.format(log_dir))
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("{}/result/{}".format(log_dir, exp_name), exist_ok=True)
        with open(f"{log_dir}/result/{exp_name}/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def caption_test_step(pl_module, batch, output, beam_size=5):
    captions = []
    tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    max_len = pl_module.hparams.config["max_text_len"]
    if not pl_module.training:
        bs = batch["text_ids"].size(0)
        text_ids = torch.ones((bs, max_len), device=batch["text_ids"].device, dtype=batch["text_ids"].dtype)
        text_ids[:, 0] = tokenizer.cls_token_id
        text_ids[:, 1:] = tokenizer.pad_token_id
        search_size = bs * beam_size
        end_seq = torch.zeros_like(text_ids[:, 0])
        end_seq = end_seq > 0
        ret = pl_module.infer_caption(batch, mask_text=False, mask_image=False)
        image_embeds = ret["image_embeds"]
        batch["text_masks"] = None
        for i in range(max_len - 1):
            batch["text_ids"] = text_ids
            if i != 0:
                infer = pl_module.infer_caption(batch, mask_text=False, mask_image=False, image_embeds=image_embeds)
            else:
                infer = ret
            mlm_logits = pl_module.mlm_score(infer["text_feats"][:, i : i + 1])
            mlm_logits[:, 0, tokenizer.mask_token_id] = -10000
            mlm_logits = torch.log_softmax(mlm_logits, dim=-1)
            if i == 0:
                tgt_prev_tokens = mlm_logits.argsort(descending=True, dim=-1)[:, :, :beam_size]
                head_logp = mlm_logits.gather(dim=-1, index=tgt_prev_tokens)

                tgt_prev_tokens = tgt_prev_tokens.permute(0, 2, 1).reshape(search_size, 1).contiguous()
                head_logp = head_logp.view(search_size, 1)
                head_lengths = torch.ones_like(head_logp)
                text_ids = text_ids.view(bs, 1, -1).repeat(1, beam_size, 1).view(search_size, -1)
                end_seq = end_seq.view(bs, 1, -1).repeat(1, beam_size, 1).view(search_size, 1)
                scores = torch.zeros_like(end_seq)
                padded = torch.full((search_size, 1), 1, dtype=torch.long, device=text_ids.device)

                hs = image_embeds.size(-1)
                image_embeds = image_embeds.view(bs, 1, -1, hs).repeat(1, beam_size, 1, 1).view(search_size, -1, hs)
                text_ids[:, i + 1] = tgt_prev_tokens.view(-1)
                end_seq = (tgt_prev_tokens == tokenizer.sep_token_id) | (tgt_prev_tokens == tokenizer.pad_token_id)
            else:
                decoder_lengths = 1.0 - end_seq.to(mlm_logits.dtype)
                mlm_logits = mlm_logits * decoder_lengths[:, :, None]
                mlm_logits = mlm_logits + head_logp[:, :, None]
                mlm_logits = mlm_logits.view(bs, beam_size, 1, -1).permute(0, 2, 1, 3)
                vocab_size = mlm_logits.size(3)
                decoder_lengths = decoder_lengths + head_lengths
                decoder_lengths = decoder_lengths.view(bs, beam_size, 1).permute(0, 2, 1)
                decoder_normed_logp = (
                    (mlm_logits / (decoder_lengths[:, :, :, None] + 1e-9)).contiguous().view(bs, 1, -1)
                )
                decoder_logp = mlm_logits.contiguous().view(bs, 1, -1)
                top_idx = decoder_normed_logp.argsort(descending=True, dim=-1)[:, :, :beam_size]
                top_logp = decoder_logp.gather(dim=-1, index=top_idx)
                top_tokens = top_idx % vocab_size
                top_prev_idx = top_idx // vocab_size
                top_prev_idx += torch.arange(bs, dtype=torch.long, device=mlm_logits.device)[:, None, None] * beam_size

                top_prev_idx = top_prev_idx.permute(0, 2, 1)
                top_prev_idx = top_prev_idx.contiguous().view(search_size, 1)
                top_logp = top_logp.permute(0, 2, 1)
                head_logp = top_logp.contiguous().view(search_size, 1)
                top_lengths = decoder_lengths.permute(0, 2, 1)
                head_lengths = top_lengths.contiguous().view(search_size, 1)
                top_tokens = top_tokens.permute(0, 2, 1)
                top_tokens = top_tokens.contiguous().view(search_size, 1)
                prev_ended = end_seq.gather(dim=0, index=top_prev_idx)
                tgt_prev_tokens = (1 - prev_ended.to(torch.long)) * top_tokens + prev_ended.to(torch.long) * padded
                t_size = text_ids.size(1)
                top_decoded_idx = top_prev_idx.repeat(1, t_size)
                text_ids = text_ids.gather(dim=0, index=top_decoded_idx)
                text_ids[:, i + 1] = tgt_prev_tokens.view(-1)
                end_seq = (tgt_prev_tokens == tokenizer.sep_token_id) | (tgt_prev_tokens == tokenizer.pad_token_id)
                if torch.sum(end_seq) == len(end_seq):
                    break

        text_ids = text_ids.view(bs, beam_size, -1)[:, 0, 1:]
        text_ids = text_ids.contiguous()
        text_ids[text_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id
        text_ids[text_ids == tokenizer.cls_token_id] = tokenizer.pad_token_id
        for text_id in text_ids:
            captions.append(tokenizer.decode(text_id).replace(tokenizer.pad_token, ""))

    return {"image_ids": batch["iid"], "captions": captions}


def caption_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    image_ids, captions = list(), list()
    for out in outs:
        image_ids += out["image_ids"]
        captions += out["captions"]

    rets = list()
    for i, iid in enumerate(image_ids):
        rets.append({"image_id": iid, "caption": captions[i]})
    with open(f"caption_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("caption_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        jsons_dedup = list()
        iid_dedup = set()
        for x in jsons:
            if not x["image_id"] in iid_dedup:
                jsons_dedup.append(x)
                iid_dedup.add(x["image_id"])
        os.makedirs("result", exist_ok=True)
        with open(f"result/caption.json", "w") as fp:
            json.dump(jsons_dedup, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"caption_{rank}.json")


def compute_caption_mle(pl_module, batch):
    tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    infer = pl_module.infer_caption(batch, mask_text=False, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_ids"]
    mlm_labels = torch.cat([mlm_labels[:, 1:], torch.ones_like(mlm_labels[:, :1]) * tokenizer.pad_token_id], 1)
    mlm_labels[mlm_labels == tokenizer.pad_token_id] = -100

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "caption_mle_loss": mlm_loss,
        "caption_mle_logits": mlm_logits,
        "caption_mle_labels": mlm_labels,
        "caption_mle_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_caption_mle_loss")(ret["caption_mle_loss"])
    acc = getattr(pl_module, f"{phase}_caption_mle_accuracy")(ret["caption_mle_logits"], ret["caption_mle_labels"])
    pl_module.log(f"caption_mle/{phase}/loss", loss)
    pl_module.log(f"caption_mle/{phase}/accuracy", acc)

    return ret


def compute_caption_cider(pl_module, batch, beam_size=5, alpha=0.3):
    CiderD_scorer = pl_module.cider_scorer
    tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    rl_loss = 0
    max_len = pl_module.hparams.config["max_text_len"]
    if pl_module.training:
        from collections import OrderedDict

        def _wrap_sentence(s):
            # ensure the sentence ends with <eos> token
            # in order to keep consisitent with cider_cached_tokens
            r = s.strip()
            if r.endswith("."):
                r = r[:-1]
            r += " <eos>"
            return r

        with torch.no_grad():
            captions = []
            bs = batch["text_ids"].size(0)
            text_ids = torch.ones((bs, max_len), device=batch["text_ids"].device, dtype=batch["text_ids"].dtype)
            text_ids[:, 0] = tokenizer.cls_token_id
            text_ids[:, 1:] = tokenizer.pad_token_id
            search_size = bs * beam_size
            end_seq = torch.zeros_like(text_ids[:, 0])
            end_seq = end_seq > 0

            batch_detok = {
                "image": [batch["image"][0]],
                "text_ids": text_ids[:, :2],
                "text_labels": None,
                "text_masks": (text_ids[:, :2] != tokenizer.pad_token_id),
            }
            ret = pl_module.infer_caption(batch_detok, mask_text=False, mask_image=False)
            image_embeds = ret["image_embeds"]
            batch_detok["text_masks"] = None
            for i in range(max_len - 1):
                batch_detok["text_ids"] = text_ids
                infer = (
                    pl_module.infer_caption(batch_detok, mask_text=False, mask_image=False, image_embeds=image_embeds)
                    if i != 0
                    else ret
                )
                mlm_logits = pl_module.mlm_score(infer["text_feats"][:, i : i + 1])
                mlm_logits[:, 0, tokenizer.mask_token_id] = -10000
                mlm_logits = torch.log_softmax(mlm_logits, dim=-1)
                if i == 0:
                    probs = mlm_logits.exp_()
                    tgt_prev_tokens = torch.multinomial(
                        probs.view(bs, -1),
                        beam_size,
                        replacement=True,
                    ).view(bs, 1, beam_size)
                    head_logp = mlm_logits.gather(dim=-1, index=tgt_prev_tokens)

                    tgt_prev_tokens = tgt_prev_tokens.permute(0, 2, 1).reshape(search_size, 1).contiguous()
                    head_logp = head_logp.view(search_size, 1)
                    head_lengths = torch.ones_like(head_logp)
                    text_ids = text_ids.view(bs, 1, -1).repeat(1, beam_size, 1).view(search_size, -1)
                    end_seq = end_seq.view(bs, 1, -1).repeat(1, beam_size, 1).view(search_size, 1)
                    scores = torch.zeros_like(end_seq)
                    padded = torch.full((search_size, 1), 1, dtype=torch.long, device=text_ids.device)

                    hs = image_embeds.size(-1)
                    image_embeds = image_embeds.view(bs, 1, -1, hs).repeat(1, beam_size, 1, 1).view(search_size, -1, hs)
                    text_ids[:, i + 1] = tgt_prev_tokens.view(-1)
                    end_seq = (tgt_prev_tokens == tokenizer.pad_token_id) | (tgt_prev_tokens == tokenizer.sep_token_id)
                else:
                    probs = mlm_logits.exp_()
                    top_tokens = torch.multinomial(
                        probs.view(bs * beam_size, -1),
                        1,
                        replacement=True,
                    ).view(bs * beam_size, 1)

                    decoder_lengths = 1.0 - end_seq.to(mlm_logits.dtype)
                    decoder_lengths = decoder_lengths + head_lengths
                    head_lengths = decoder_lengths
                    prev_ended = end_seq
                    tgt_prev_tokens = (1 - prev_ended.to(torch.long)) * top_tokens + prev_ended.to(torch.long) * padded
                    text_ids[:, i + 1] = tgt_prev_tokens.view(-1)
                    end_seq = (tgt_prev_tokens == tokenizer.pad_token_id) | (tgt_prev_tokens == tokenizer.sep_token_id)
                    if torch.sum(end_seq) == len(end_seq):
                        break

        text_ids = text_ids.view(bs * beam_size, -1)
        text_ids = text_ids.contiguous()
        pl_module.train()
        img_sz = batch["image"][0].size(2)
        batch_rl = {
            "image": [
                batch["image"][0]
                .view(bs, 1, 3, img_sz, img_sz)
                .repeat(1, beam_size, 1, 1, 1)
                .view(bs * beam_size, 3, img_sz, img_sz)
            ],
            "text_ids": text_ids,
            "text_labels": None,
            "text_masks": (text_ids != tokenizer.pad_token_id),
        }
        infer_rl = pl_module.infer_caption(batch_rl, mask_text=False, mask_image=False)
        rl_labels = infer_rl["text_ids"]
        rl_labels = torch.cat([rl_labels[:, 1:], torch.ones_like(rl_labels[:, :1]) * tokenizer.pad_token_id], 1)
        pad_mask = rl_labels == tokenizer.pad_token_id
        rl_logits = pl_module.mlm_score(infer_rl["text_feats"])
        rl_logits = torch.log(torch.nn.functional.softmax(rl_logits, dim=-1) + 1e-9)
        rl_bs, rl_seq_len, vocab_size = rl_logits.size()
        rl_labels = rl_labels.view(-1, 1)
        rl_logits = rl_logits.view(-1, vocab_size)
        rl_probs = rl_logits.gather(dim=-1, index=rl_labels)
        rl_probs = rl_probs.view(rl_bs, rl_seq_len)
        rl_probs[pad_mask] = 0
        rl_len = torch.sum(1 - pad_mask.float(), dim=-1)
        rl_probs = torch.sum(rl_probs, dim=-1)
        rl_probs = rl_probs / (rl_len + 1e-9)
        rl_probs = rl_probs.view(bs, -1)

        gen_res = []
        for text_id in text_ids:
            gen_res.append(
                (
                    tokenizer.decode(text_id)
                    .replace(tokenizer.cls_token, "")
                    .replace(tokenizer.sep_token, "")
                    .replace(tokenizer.pad_token, "")
                )
            )
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [_wrap_sentence(gen_res[i])]

        gt_res = []
        for gt in batch["gt_txt"]:
            for _ in range(beam_size):
                gt_res.append(gt)
        gts = OrderedDict()
        gt_res_ = [[_wrap_sentence(gt_res[i][j]) for j in range(len(gt_res[i]))] for i in range(len(gt_res))]
        for i in range(gen_res_size):
            gts[i] = gt_res_[i]

        res_ = [{"image_id": i, "caption": res[i]} for i in range(len(res))]
        _, batch_cider_scores = CiderD_scorer.compute_score(gts, res_)

        rl_loss = rl_probs.view(-1) * (
            100.0 - 100.0 * torch.tensor(batch_cider_scores, device=rl_probs.device).view(-1)
        )
        rl_loss = torch.sum(rl_loss) / bs

    caption_infer = pl_module.infer_caption(batch, mask_text=False, mask_image=False)
    caption_logits = pl_module.mlm_score(caption_infer["text_feats"])

    caption_labels = caption_infer["text_ids"]
    caption_labels = torch.cat(
        [caption_labels[:, 1:], torch.ones_like(caption_labels[:, :1]) * tokenizer.pad_token_id], 1
    )
    pad_mask = caption_labels == tokenizer.pad_token_id

    caption_labels[pad_mask] = -100
    caption_loss = F.cross_entropy(
        caption_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        caption_labels.view(-1),
        ignore_index=-100,
    )
    mlm_loss = alpha * caption_loss + (1 - alpha) * rl_loss
    caption_ids = caption_infer["text_ids"]

    ret = {
        "caption_cider_loss": mlm_loss,
        "caption_cider_logits": caption_logits,
        "caption_cider_labels": caption_labels,
        "caption_cider_ids": caption_ids,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_caption_cider_loss")(ret["caption_cider_loss"])
    acc = getattr(pl_module, f"{phase}_caption_cider_accuracy")(
        ret["caption_cider_logits"], ret["caption_cider_labels"]
    )
    pl_module.log(f"caption_cider/{phase}/loss", loss)
    pl_module.log(f"caption_cider/{phase}/accuracy", acc)

    return ret
