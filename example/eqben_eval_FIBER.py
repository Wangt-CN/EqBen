import torch
import copy
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import eqben.eqben_utils as eqben_utils
import eqben.eqben_acc as eqben_acc


##################### 0. IMPORT USER-MODEL PACKAGE ###########################
# In this template, we use "USER_MODEL" to represent the VLP model which would be evaluated.
# Please check other files within the example folder for the specific VLP model.
from fiber.config import ex
from fiber.modules import FIBERTransformerSS
from fiber.datamodules.datamodule_base import get_pretrained_tokenizer
from fiber.transforms import (
    albef_transform,
    albef_transform_randaug,
)

_transforms = {
    "albef": albef_transform,
    "albef_randaug": albef_transform_randaug,
}

##################### 1. DATA SET PATH ###########################
# Please identify the path of "img_root" and "ann_root" in the server.

DATASET_INFO = {
    "eqben": {"func": eqben_utils.EqBenALL,
              "img_root": '/path/to/eqben/image',
              "ann_root": '/path/to/eqben/annotation/ann_json_finegrained_random.json'},
    'winoground': {"func": eqben_utils.Winoground,
                   'img_root':'/path/to/winoground/images',
                   'ann_root':'/path/to/winoground/examples.jsonl'},
    'valse': {"func": eqben_utils.VALSE,
              'img_root':'/path/to/valse/valse_ann/ours/img',
              'ann_root':'/path/to/valse/valse_ann/ours/global_ann.json'}
}



################ 2. CUSTOMIZED DATA TOOL #####################
# In USER_MODEL, how will the data (image/caption) be pre-processed?
# How will the data be collated in the batch?

class customized_data_toolkit():
    def __init__(self, config):
        self.config = config
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])

    def process_img_pixel(self, image):
        # Define the model-specifc data pre-process (e.g., augmentation) for image pixel.
        image = Image.open(image).convert("RGB")
        return _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image)

    def process_img_npy(self, image):
        # Define the model-specifc data pre-process (e.g., augmentation) for image numpy file (Youcook2).
        image = Image.fromarray(np.load(image)[:, :, [2, 1, 0]], 'RGB')
        return _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image)

    def get_img_batch(self, img_batch):
        batch_size = len(img_batch)
        # get max size
        img_sizes = list()
        img_sizes += [i.shape for i in img_batch]

        max_height, max_width = max([i[1] for i in img_sizes]), max([i[2] for i in img_sizes])
        new_img_batch = torch.zeros(batch_size, 3, max_height, max_width)
        for bi in range(batch_size):
            orig = img_batch[bi]
            new_img_batch[bi, :, :orig.shape[1], :orig.shape[2]] = orig
        return new_img_batch

    def collate(self, batch):
        # Define the model-specific data collate method, will be forwarded to "collate_fn"
        image0_batch, image1_batch = [item[0] for item in batch], [item[1] for item in batch]
        caption0_batch, caption1_batch = [item[2] for item in batch], [item[3] for item in batch]

        image0_batch_collate = self.get_img_batch(image0_batch)
        image1_batch_collate = self.get_img_batch(image1_batch)

        encode0_batch_collate = self.tokenizer(caption0_batch, padding="max_length",
                                              truncation=True, max_length=40, # config['max_text_len']
                                              return_tensors="pt",
                                              return_special_tokens_mask=True)

        encode1_batch_collate = self.tokenizer(caption1_batch, padding="max_length",
                                              truncation=True, max_length=40, # config['max_text_len']
                                              return_tensors="pt",
                                              return_special_tokens_mask=True)

        return image0_batch_collate, image1_batch_collate, encode0_batch_collate, encode1_batch_collate, caption0_batch, caption1_batch

    def collate_eqben(self, batch):
        # Define the model-specific data collate method, will be forwarded to "collate_fn"
        image_batch = [item[0] for item in batch]
        caption_batch = [item[1] for item in batch]

        image_batch_collate = self.get_img_batch(image_batch)

        encode_batch_collate = self.tokenizer(caption_batch, padding="max_length",
                                              truncation=True, max_length=40, # config['max_text_len']
                                              return_tensors="pt",
                                              return_special_tokens_mask=True)


        return image_batch_collate, encode_batch_collate, caption_batch



################ 3. USER_MODEL INFERENCE PROCESS #####################
# For USER_MODEL, how to forward the data (image/caption) and obtain the image-text simiarity during the inference stage.

def _loss_names(d):
    ret = {
        "itm": 0.5,
        "mlm": 0.,
        "itc":0.5,
        "vqa": 0,
        "nlvr2": 0,
        "caption_mle": 0,
        "caption_gold": 0,
        "caption_cider": 0,
    }
    ret.update(d)
    return ret


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    loss_names = _loss_names({"itm": 0.5})
    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    # 3.1. SETUP LOGGER
    eval_results_path = os.path.basename(_config["load_path"])
    os.makedirs(os.path.join(_config["eval_root"], eval_results_path), exist_ok=True)

    from eqben.eqben_logger import LOGGER as logger_txt, add_log_to_file
    logger_path = "{}/{}/eval_{}_{}_log.txt".format(_config["eval_root"], eval_results_path, _config["eval_task"], _config["eval_data"])
    logger_txt.info("creating log at: {}".format(logger_path))
    add_log_to_file(logger_path)


    # 3.2. DATALOADER/USER_MODEL SETUP
    eval_dataset = DATASET_INFO[_config["eval_data"]]["func"](img_root=DATASET_INFO[_config["eval_data"]]["img_root"], ann_root=DATASET_INFO[_config["eval_data"]]["ann_root"], config=_config, customized_data_toolkit=customized_data_toolkit(_config))
    gpu_cnt = torch.cuda.device_count()
    print('use {} GPUs; '.format(gpu_cnt))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=int(gpu_cnt*128), num_workers=0, shuffle=False, collate_fn=eval_dataset.collate)

    device = "cuda"
    model = FIBERTransformerSS(_config)
    model.setup("test")
    model.eval()
    model = torch.nn.DataParallel(model).to(device)


    # 3.3. FORWARD PROCESS

    def infer_batch(batch):

        with torch.no_grad():
            scores = {}
            infer = model(batch)
            itm_logits = model.module.itm_score(infer["cls_feats"])
            scores['itm_score_logits'] = itm_logits[:, 1]
            score = torch.nn.functional.softmax(itm_logits)[:, 1]
            scores['itm_score'] = score
        return scores


    eval_root_dir = _config["eval_root"] if _config["eval_root"] is not None else os.path.join('result', _config["exp_name"])
    if not os.path.exists(os.path.join(eval_root_dir, eval_results_path)):
        os.makedirs(os.path.join(eval_root_dir, eval_results_path))


    if _config["eval_data"] == "eqben":
        fiber_score_all = []
        for idx, (image, encode, caption) in enumerate(tqdm(eval_dataloader)):
            image, encode = image.to(device), encode.to(device)
            batch_c_i = {"text": caption, "image": [image], "text_ids": encode["input_ids"], "text_labels": encode["input_ids"], "text_masks": encode["attention_mask"]}
            fiber_score = infer_batch(batch_c_i)
            fiber_score_all.append(fiber_score["itm_score"].cpu())

        fiber_score_all = torch.cat(fiber_score_all, dim=0)
        np.save('{}/{}/eqben_scores.npy'.format(_config["eval_root"], eval_results_path), fiber_score_all)

    else: # winoground / valse
        fiber_score_c0_i0_all_itm, fiber_score_c1_i0_all_itm, fiber_score_c0_i1_all_itm, fiber_score_c1_i1_all_itm = [], [], [], []
        for idx, (image0, image1, encode0, encode1, caption0, caption1) in enumerate(tqdm(eval_dataloader)):

            image0, image1, encode0, encode1 = image0.to(device), image1.to(device), encode0.to(device), encode1.to(device)
            batch_c0_i0 = {"text":caption0, "image":[image0], "text_ids":encode0["input_ids"], "text_labels":encode0["input_ids"], "text_masks":encode0["attention_mask"]}
            batch_c1_i0 = {"text":caption1, "image":[image0], "text_ids":encode1["input_ids"], "text_labels":encode1["input_ids"], "text_masks":encode1["attention_mask"]}
            batch_c0_i1 = {"text":caption0, "image":[image1], "text_ids":encode0["input_ids"], "text_labels":encode0["input_ids"], "text_masks":encode0["attention_mask"]}
            batch_c1_i1 = {"text":caption1, "image":[image1], "text_ids":encode1["input_ids"], "text_labels":encode1["input_ids"], "text_masks":encode1["attention_mask"]}

            fiber_score_c0_i0_dict = infer_batch(batch_c0_i0)
            fiber_score_c1_i0_dict = infer_batch(batch_c1_i0)
            fiber_score_c0_i1_dict = infer_batch(batch_c0_i1)
            fiber_score_c1_i1_dict = infer_batch(batch_c1_i1)

            fiber_score_c0_i0_all_itm.append(fiber_score_c0_i0_dict["itm_score"].cpu())
            fiber_score_c1_i0_all_itm.append(fiber_score_c1_i0_dict["itm_score"].cpu())
            fiber_score_c0_i1_all_itm.append(fiber_score_c0_i1_dict["itm_score"].cpu())
            fiber_score_c1_i1_all_itm.append(fiber_score_c1_i1_dict["itm_score"].cpu())



        fiber_score_c0_i0_all_itm = torch.cat(fiber_score_c0_i0_all_itm, dim=0)
        fiber_score_c1_i0_all_itm = torch.cat(fiber_score_c1_i0_all_itm, dim=0)
        fiber_score_c0_i1_all_itm = torch.cat(fiber_score_c0_i1_all_itm, dim=0)
        fiber_score_c1_i1_all_itm = torch.cat(fiber_score_c1_i1_all_itm, dim=0)
        eval_fiber_scores_itm = {'raw_info':eval_dataset.sample_pair, "c0_i0": fiber_score_c0_i0_all_itm, "c0_i1": fiber_score_c0_i1_all_itm, "c1_i0": fiber_score_c1_i0_all_itm, "c1_i1": fiber_score_c1_i1_all_itm, "model_path": _config["load_path"]}



################ 4. CALCULATE/SAVE THE PERFORMANCE #####################

    if 'valse' in _config["eval_data"]:
        score_valse_acc, score_valse_min, score_valse_pair = eqben_acc.cal_valse_score(eval_fiber_scores_itm)
        logger_txt.info("============ ITM Score =============")
        logger_txt.info("valse score pair: {}".format(score_valse_pair))
        logger_txt.info("valse score min(pc,pf): {}".format(score_valse_min))
        logger_txt.info("valse score acc: {}".format(score_valse_acc))

    else:
        text_score_itm, image_score_itm, group_score_itm = eqben_acc.cal_wino_score(eval_fiber_scores_itm)
        eval_fiber_scores_itm["scores"] = {"text_score":text_score_itm, "image_score":image_score_itm, "group_score":group_score_itm}
        logger_txt.info("============ ITM Score =============")
        logger_txt.info("text score: {}".format(text_score_itm))
        logger_txt.info("image score: {}".format(image_score_itm))
        logger_txt.info("group score: {}".format(group_score_itm))


