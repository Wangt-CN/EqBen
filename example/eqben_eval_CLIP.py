import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import eqben.eqben_utils as eqben_utils
import eqben.eqben_acc as eqben_acc


##################### 0. IMPORT USER-MODEL PACKAGE ###########################
# In this template, we use "USER_MODEL" to represent the VLP model which would be evaluated.
# Please check other files within the example folder for the specific VLP model.
import clip



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
    def __init__(self, transform):
        self.transform = transform

    def process_img_pixel(self, image):
        # Define the model-specifc data pre-process (e.g., augmentation) for image pixel.
        image = Image.open(image).convert("RGB")
        return self.transform(image)

    def process_img_npy(self, image):
        # Define the model-specifc data pre-process (e.g., augmentation) for image numpy file (Youcook2).
        image = Image.fromarray(np.load(image)[:, :, [2, 1, 0]], 'RGB')
        return self.transform(image)



################ 3. USER_MODEL INFERENCE PROCESS #####################
# For USER_MODEL, how to forward the data (image/caption) and obtain the image-text simiarity during the inference stage.

def main(config):

    # 3.1. SETUP LOGGER
    load_file_basename = 'clip_rn50'
    eval_results_path = 'EqBen_results/{}'.format(load_file_basename)
    os.makedirs(eval_results_path, exist_ok=True)

    from eqben.eqben_logger import LOGGER as logger_txt, add_log_to_file
    logger_path = "{}/eval_{}_{}_log.txt".format(eval_results_path, config.eval_task, config.eval_data)
    logger_txt.info("creating log at: {}".format(logger_path))
    add_log_to_file(logger_path)



    # 3.2. DATALOADER/USER_MODEL SETUP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)
    model = model.to(device).eval()

    eval_dataset = DATASET_INFO[config.eval_data]["func"](img_root=DATASET_INFO[config.eval_data]["img_root"], ann_root=DATASET_INFO[config.eval_data]["ann_root"], config=config, customized_data_toolkit=customized_data_toolkit(preprocess))
    gpu_cnt = torch.cuda.device_count()
    print('use {} GPUs; '.format(gpu_cnt))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=int(gpu_cnt*128), num_workers=0, shuffle=False, collate_fn=eval_dataset.collate)
    clip_score_c0_i0_all, clip_score_c1_i0_all, clip_score_c0_i1_all, clip_score_c1_i1_all = [], [], [], []



    # 3.3. FORWARD PROCESS

    def clip_forward_batch_singlepair(model, image, caption):
        with torch.no_grad():
            text_encoded = model.encode_text(caption)
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
            img_feature = model.encode_image(image)
            img_feature /= img_feature.norm(dim=-1, keepdim=True)

            sim = (text_encoded * img_feature).sum(dim=-1)
            return sim

    def clip_forward_batch(model, image0, image1, caption0, caption1):
        with torch.no_grad():
            text_encoded0 = model.encode_text(caption0)
            text_encoded0 /= text_encoded0.norm(dim=-1, keepdim=True)
            text_encoded1 = model.encode_text(caption1)
            text_encoded1 /= text_encoded1.norm(dim=-1, keepdim=True)

            img_feature0 = model.encode_image(image0)
            img_feature0 /= img_feature0.norm(dim=-1, keepdim=True)
            img_feature1 = model.encode_image(image1)
            img_feature1 /= img_feature1.norm(dim=-1, keepdim=True)

            sim_c0_i0 = (text_encoded0 * img_feature0).sum(dim=-1)
            sim_c0_i1 = (text_encoded0 * img_feature1).sum(dim=-1)
            sim_c1_i0 = (text_encoded1 * img_feature0).sum(dim=-1)
            sim_c1_i1 = (text_encoded1 * img_feature1).sum(dim=-1)

            return sim_c0_i0, sim_c1_i0, sim_c0_i1, sim_c1_i1


    if config.eval_data == "eqben":
        clip_score_all = []
        for idx, (image, caption) in enumerate(tqdm(eval_dataloader)):
            image = image.to(device)
            caption = clip.tokenize(caption).to(device)
            clip_score = clip_forward_batch_singlepair(model, image, caption)
            clip_score_all.append(clip_score.cpu())

        clip_score_all = torch.cat(clip_score_all, dim=0)
        np.save('{}/eqben_scores.npy'.format(eval_results_path), clip_score_all)


    else:
        for idx, (image0, image1, caption0, caption1) in enumerate(tqdm(eval_dataloader)):
            image0, image1 = image0.to(device), image1.to(device)
            caption0, caption1 = clip.tokenize(caption0).to(device), clip.tokenize(caption1).to(device)
            clip_score_c0_i0, clip_score_c1_i0, clip_score_c0_i1, clip_score_c1_i1 = clip_forward_batch(model, image0, image1, caption0, caption1)

            clip_score_c0_i0_all.append(clip_score_c0_i0.cpu())
            clip_score_c1_i0_all.append(clip_score_c1_i0.cpu())
            clip_score_c0_i1_all.append(clip_score_c0_i1.cpu())
            clip_score_c1_i1_all.append(clip_score_c1_i1.cpu())

        clip_score_c0_i0_all = torch.cat(clip_score_c0_i0_all, dim=0)
        clip_score_c1_i0_all = torch.cat(clip_score_c1_i0_all, dim=0)
        clip_score_c0_i1_all = torch.cat(clip_score_c0_i1_all, dim=0)
        clip_score_c1_i1_all = torch.cat(clip_score_c1_i1_all, dim=0)
        eval_clip_scores_itm = {'raw_info':eval_dataset.sample_pair, "c0_i0": clip_score_c0_i0_all, "c0_i1": clip_score_c0_i1_all, "c1_i0": clip_score_c1_i0_all, "c1_i1": clip_score_c1_i1_all}


        ################ 4. CALCULATE/SAVE THE PERFORMANCE #####################
        if 'valse' in config.eval_data:
            score_valse_acc, score_valse_min, score_valse_pair = eqben_acc.cal_valse_score(eval_clip_scores_itm)
            logger_txt.info("============ ITM Score =============")
            logger_txt.info("valse score pair: {}".format(score_valse_pair))
            logger_txt.info("valse score min(pc,pf): {}".format(score_valse_min))
            logger_txt.info("valse score acc: {}".format(score_valse_acc))

        else:
            text_score_itm, image_score_itm, group_score_itm = eqben_acc.cal_wino_score(eval_clip_scores_itm)
            eval_clip_scores_itm["scores"] = {"text_score":text_score_itm, "image_score":image_score_itm, "group_score":group_score_itm}
            logger_txt.info("============ ITM Score =============")
            logger_txt.info("text score: {}".format(text_score_itm))
            logger_txt.info("image score: {}".format(image_score_itm))
            logger_txt.info("group score: {}".format(group_score_itm))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', default='eqbenag')
    parser.add_argument('--eval_task', default='clip_pretrian')
    config = parser.parse_args()

    main(config)