import torch
import copy
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import eqben.eqben_utils as eqben_utils
import eqben.eqben_acc as eqben_acc


##################### 0. IMPORT USER-MODEL PACKAGE ###########################
# In this template, we use "USER_MODEL" to represent the VLP model which would be evaluated.
# Please check other files within the example folder for the specific VLP model.




##################### 1. DATA SET PATH ###########################
# Please identify the path of "img_root" and "ann_root" in the server.

DATASET_INFO = {
    "eqben": {"func": eqben_utils.EqBenALL,
              "img_root": '',
              "ann_root": ''},
    'winoground': {"func": eqben_utils.Winoground,
                   'img_root':'',
                   'ann_root':''},
    'valse': {"func": eqben_utils.VALSE,
              'img_root':'',
              'ann_root':''}
}



################ 2. CUSTOMIZED DATA TOOL #####################
# In USER_MODEL, how will the data (image/caption) be pre-processed?
# How will the data be collated in the batch?


class customized_data_toolkit():
    def __init__(self):
        pass

    def process_img_pixel(self, image):
        # Define the model-specifc data pre-process (e.g., augmentation) for image pixel.
        pass

    def process_img_npy(self, image):
        # Define the model-specifc data pre-process (e.g., augmentation) for image numpy file (Youcook2).
        pass

    def collate(self, batch):
        # Define the model-specific data collate method, will be forwarded to "collate_fn"
        pass




################ 3. USER_MODEL INFERENCE PROCESS #####################
# For USER_MODEL, how to forward the data (image/caption) and obtain the image-text simiarity during the inference stage.
config = {"eval_data": "eqbenag"}


# 3.1. SETUP LOGGER
eval_results_path = ''
os.makedirs(os.path.join(eval_results_path), exist_ok=True)
from eqben.eqben_logger import LOGGER as logger_txt, add_log_to_file
logger_path = '{}/xxx'.format(eval_results_path)
logger_txt.info("creating log at: {}".format(logger_path))
add_log_to_file(logger_path)


# 3.2. DATALOADER/USER_MODEL SETUP
eval_dataset = DATASET_INFO[config["eval_data"]]["func"](img_root=DATASET_INFO[config["eval_data"]]["img_root"], ann_root=DATASET_INFO[config["eval_data"]]["ann_root"], config=config,
                                                          customized_data_toolkit=customized_data_toolkit()) # automatic create PyTorch dataset for the evaluated data
gpu_cnt = torch.cuda.device_count()
print('use {} GPUs; '.format(gpu_cnt))
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=int(gpu_cnt * 128), num_workers=int(gpu_cnt*4), shuffle=False, collate_fn=eval_dataset.collate)

USER_MODEL = None


# 3.3. FORWARD PROCESS
if config.eval_data == "eqben": # for eqben, please send the result file to the server
    score_all = []
    for idx, (image, caption) in enumerate(tqdm(eval_dataloader)):
        score = USER_MODEL(image, caption)
        score_all.append(score.cpu())
    score_all = torch.cat(score_all, dim=0)
    np.save('{}/eqben_scores.npy'.format(eval_results_path), score_all)


else: # for winoground/valse, directly calculate
    score_c0_i0_all, score_c1_i0_all, score_c0_i1_all, score_c1_i1_all = [], [], [], []
    for idx, (image0, image1, caption0, caption1) in enumerate(tqdm(eval_dataloader)):
        score_c0_i0, score_c1_i0, score_c0_i1, score_c1_i1 = USER_MODEL(image0, image1, caption0, caption1)
        score_c0_i0_all.append(score_c0_i0)
        score_c1_i0_all.append(score_c1_i0)
        score_c0_i1_all.append(score_c0_i1)
        score_c1_i1_all.append(score_c1_i1)
    score_c0_i0_all = torch.cat(score_c0_i0_all, dim=0)
    score_c1_i0_all = torch.cat(score_c1_i0_all, dim=0)
    score_c0_i1_all = torch.cat(score_c0_i1_all, dim=0)
    score_c1_i1_all = torch.cat(score_c1_i1_all, dim=0)
    eval_scores_dict = {'raw_info':eval_dataset.sample_pair, "c0_i0": score_c0_i0_all, "c0_i1": score_c0_i1_all, "c1_i0": score_c1_i0_all, "c1_i1": score_c1_i1_all, "model_path": config["load_path"]}



    ################ 4. CALCULATE/SAVE THE PERFORMANCE #####################

    if 'valse' in config["eval_data"]:
        score_valse_acc, score_valse_min, score_valse_pair = eqben_acc.cal_valse_score(eval_scores_dict)
        logger_txt.info("============ Score =============")
        logger_txt.info("valse score pair: {}".format(score_valse_pair))
        logger_txt.info("valse score min(pc,pf): {}".format(score_valse_min))
        logger_txt.info("valse score acc: {}".format(score_valse_acc))

    else:
        text_score_itm, image_score_itm, group_score_itm = eqben_acc.cal_wino_score(eval_scores_dict)
        eval_scores_dict["scores"] = {"text_score": text_score_itm, "image_score": image_score_itm, "group_score": group_score_itm}
        logger_txt.info("============ Score =============")
        logger_txt.info("text score: {}".format(text_score_itm))
        logger_txt.info("image score: {}".format(image_score_itm))
        logger_txt.info("group score: {}".format(group_score_itm))


