import torch
# torch.multiprocessing.set_start_method('spawn')
import copy
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

from fiber.config import ex
from fiber.modules import FIBERTransformerSS

from fiber.datamodules.datamodule_base import get_pretrained_tokenizer

from fiber.transforms import (
    square_transform,
    square_transform_randaug,
)

_transforms = {
    "square": square_transform,
    "square_randaug": square_transform_randaug,
}


_data_info = {
    ### ours benchmark
    "my_VG": {"img_root": '/vigstandard_data/v-tanwang/code/video_benchmark/vg/ours_benchmark/20220720_v1.0/frames', "ann_root": '/vigstandard_data/v-tanwang/code/video_benchmark/vg/ours_benchmark/20220720_v1.0/global_ann.json'},
    "my_VG_2.0": {"img_root": '/vigstandard_data/v-tanwang/code/video_benchmark/vg/ours_benchmark/20220823_v2.0/frames',
              "ann_root": '/vigstandard_data/v-tanwang/code/video_benchmark/vg/ours_benchmark/20220823_v2.0/global_ann.json'},
    "my_youcook2": {"img_root": '/vigstandard_data/v-tanwang/code/video_benchmark/youcook2/ours_benchmark/20220710', "ann_root": '/vigstandard_data/v-tanwang/code/video_benchmark/youcook2/ours_benchmark/20220714/trainval_global_ann_filter_face'},
    # "my_youcook2": {"img_root": '/home1/wangtan/data/youcook2/ours_benchmark/20220710', "ann_root": '/home1/wangtan/data/youcook2/ours_benchmark/20220714/trainval_global_ann_filter_face'},
    "my_gebc": {"img_root": '/vigstandard_data/v-tanwang/code/video_benchmark/GEBC_data/20220804/trainval', "ann_root": '/vigstandard_data/v-tanwang/code/video_benchmark/GEBC_data/20220804/gebc_filtered_start0_end3728_json.json'},
    'my_kubric_location': {'img_root':'/vigstandard_data/v-tanwang/code/video_benchmark/kubric/our_benchmark/v0.2/location', 'ann_root':'/vigstandard_data/v-tanwang/code/video_benchmark/kubric/our_benchmark/v0.2/location/global_ann_2k.json'},
    'my_kubric_counting': {'img_root':'/vigstandard_data/v-tanwang/code/video_benchmark/kubric/our_benchmark/v0.2/counting', 'ann_root':'/vigstandard_data/v-tanwang/code/video_benchmark/kubric/our_benchmark/v0.2/counting/global_ann_2k.json'},
    'my_kubric_attribute': {'img_root':'/vigstandard_data/v-tanwang/code/video_benchmark/kubric/our_benchmark/v0.2/attribute', 'ann_root':'/vigstandard_data/v-tanwang/code/video_benchmark/kubric/our_benchmark/v0.2/attribute/global_ann_2k.json'},
    ### winoground
    'winoground': {'img_root':'/vigstandard_data/v-tanwang/code/video_benchmark/winoground/images/',
                   'ann_root':'/vigstandard_data/v-tanwang/code/video_benchmark/winoground/examples.jsonl'},
    'my_generation': {'img_root':'/vigstandard_data/v-tanwang/code/video_benchmark/my_SD/1020_v1/',
                   'ann_root':'/vigstandard_data/v-tanwang/code/video_benchmark/my_SD/global_ann_filtered.json'},
    'valse': {'img_root':'/vigstandard_data/v-tanwang/code/video_benchmark/valse/ours/img',
              'ann_root':'/vigstandard_data/v-tanwang/code/video_benchmark/valse/ours/global_ann.json'}
}


# for quad 2
# _data_info = {
#     "my_VG": {"img_root": '/datadrive_d/wangtan/azure_storage/vigstandard_data/v-tanwang/code/video_benchmark/vg/ours_benchmark/20220720_v1.0/frames', "ann_root": '/datadrive_d/wangtan/azure_storage/vigstandard_data/v-tanwang/code/video_benchmark/vg/ours_benchmark/20220720_v1.0/global_ann.json'},
#     "my_youcook2": {"img_root": '/datadrive_d/wangtan/local_file/ms_intern_proj/video_benchmark/youcook2/ours_benchmark/20220710', "ann_root": '/datadrive_d/wangtan/local_file/ms_intern_proj/video_benchmark/youcook2/ours_benchmark/20220714/trainval_global_ann_filter_face'},
#     "my_gebc": {"img_root": '/home1/wangtan/data/GEBC/our_benchmark/20220804/trainval',
#                 "ann_root": '/home1/wangtan/data/GEBC/our_benchmark/20220804/gebc_filtered_start0_end3728_json.json'}
# }

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



class Eval_METER(Dataset):
    def __init__(self):
        pass

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
        image0_batch, image1_batch = [item[0] for item in batch], [item[1] for item in batch]
        caption0_batch, caption1_batch = [item[2] for item in batch], [item[3] for item in batch]

        image0_batch_collate = self.get_img_batch(image0_batch)
        image1_batch_collate = self.get_img_batch(image1_batch)

        encode0_batch_collate =self.tokenizer(caption0_batch, padding="max_length",
                                              truncation=True, max_length=40, # config['max_text_len']
                                              return_tensors="pt",
                                              return_special_tokens_mask=True)

        encode1_batch_collate =self.tokenizer(caption1_batch, padding="max_length",
                                              truncation=True, max_length=40, # config['max_text_len']
                                              return_tensors="pt",
                                              return_special_tokens_mask=True)

        return image0_batch_collate, image1_batch_collate, encode0_batch_collate, encode1_batch_collate, caption0_batch, caption1_batch



class VisualGenome_My_METER(Eval_METER): # adapte to v1.0 version
    def __init__(self, img_root, ann_root, config):
        super(VisualGenome_My_METER, self).__init__()
        self.config = config
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(self.global_ann.items()):
            # if process_idx == 100:
            #     break ##debug
            print('processed {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            for subject_dir, subject_dir_values in video_dir_values.items():
                for frame_info in subject_dir_values:
                    image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
                    caption0, caption1 = frame_info['caption0'][0], frame_info['caption1'][0] # only use the first caption in the caption list
                    added_sample = {'subject':subject_dir, 'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
                    self.sample_pair.append(added_sample)


    def __getitem__(self, index):
        subject, image0, caption0, image1, caption1 = self.sample_pair[index]['subject'], self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = Image.open(image0).convert("RGB"), Image.open(image1).convert("RGB")
        image0, image1 = _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image0), _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)



class Youcook2_My_METER(Eval_METER):
    def __init__(self, img_root, ann_root, config):
        super(Youcook2_My_METER, self).__init__()
        self.config = config
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(self.global_ann.items()):
            print('processed {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            for frame_info in video_dir_values:
                image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
                caption0, caption1 = frame_info["caption0"], frame_info["caption1"]
                added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
                self.sample_pair.append(added_sample)


    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = Image.fromarray(np.load(image0)[:,:,[2,1,0]], 'RGB'), Image.fromarray(np.load(image1)[:,:,[2,1,0]], 'RGB')
        image0, image1 = _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image0), _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image1)
        return image0, image1, caption0, caption1


    def __len__(self):
        return len(self.sample_pair)




class GEBC_My_METER(Eval_METER):
    def __init__(self, img_root, ann_root, config):
        super(GEBC_My_METER, self).__init__()
        self.config = config
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(self.global_ann.items()):
            print('processed {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            for frame_info in video_dir_values:
                image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
                caption0, caption1 = frame_info["caption0"], frame_info["caption1"]
                added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
                self.sample_pair.append(added_sample)


    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = Image.open(image0).convert("RGB"), Image.open(image1).convert("RGB")
        image0, image1 = _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image0), _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image1)
        return image0, image1, caption0, caption1


    def __len__(self):
        return len(self.sample_pair)



class Kubric_My_METER(Eval_METER):
    def __init__(self, img_root, ann_root, config):
        super(Kubric_My_METER, self).__init__()
        self.config = config
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        # import pdb; pdb.set_trace()
        for process_idx, frame_info in enumerate(self.global_ann):
            print('processed {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
            caption0, caption1 = frame_info['caption0'], frame_info['caption1']
            added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
            self.sample_pair.append(added_sample)


    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = Image.open(image0).convert("RGB"), Image.open(image1).convert("RGB")
        image0, image1 = _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image0), _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image1)
        return image0, image1, caption0, caption1


    def __len__(self):
        return len(self.sample_pair)



class Winoground_My_METER(Eval_METER):
    def __init__(self, img_root, ann_root, config):
        super(Winoground_My_METER, self).__init__()
        self.config = config
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        self.sample_pair = []
        self.global_ann = [json.loads(example_json) for example_json in open(ann_root).readlines()]
        # import pdb; pdb.set_trace()
        for process_idx, frame_info in enumerate(self.global_ann):
            print('processed {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            image0_path, image1_path = os.path.join(img_root, frame_info['image_0']+'.png'), os.path.join(img_root, frame_info['image_1']+'.png')
            caption0, caption1 = frame_info['caption_0'], frame_info['caption_1']
            added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
            self.sample_pair.append(added_sample)


    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = Image.open(image0).convert("RGB"), Image.open(image1).convert("RGB")
        image0, image1 = _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image0), _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image1)
        return image0, image1, caption0, caption1


    def __len__(self):
        return len(self.sample_pair)



class Generation_My_METER(Eval_METER):
    def __init__(self, img_root, ann_root, config):
        super(Generation_My_METER, self).__init__()
        self.config = config
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))

        # import pdb; pdb.set_trace()
        for process_idx, frame_info in enumerate(self.global_ann):
            print('processed {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
            caption0, caption1 = frame_info['caption0'], frame_info['caption1']
            added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
            self.sample_pair.append(added_sample)


    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = Image.open(image0).convert("RGB"), Image.open(image1).convert("RGB")
        image0, image1 = _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image0), _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image1)
        return image0, image1, caption0, caption1


    def __len__(self):
        return len(self.sample_pair)


class VALSE_My_METER(Eval_METER):
    def __init__(self, img_root, ann_root, config):
        super(VALSE_My_METER, self).__init__()
        self.config = config
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        # import pdb; pdb.set_trace()
        for process_idx, frame_info in enumerate(self.global_ann):
            print('processed {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            image0_path, image1_path = os.path.join(img_root, frame_info['image_0']), os.path.join(img_root, frame_info['image_1'])
            caption0, caption1 = frame_info['caption_0'], frame_info['caption_1']
            added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
            self.sample_pair.append(added_sample)


    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = Image.open(image0).convert("RGB"), Image.open(image1).convert("RGB")
        image0, image1 = _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image0), _transforms[self.config["val_transform_keys"][0]](size=self.config["image_size"])(image1)
        return image0, image1, caption0, caption1


    def __len__(self):
        return len(self.sample_pair)



@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    if  _config["eval_task"] == 'pretrain':
        loss_names = _loss_names({"itm": 0.5})
    else:
        loss_names = _loss_names({})

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    # setup logger
    load_file_basename = os.path.basename(_config["load_path"])
    eval_results_path = 'eval_results/{}'.format(load_file_basename)
    os.makedirs(os.path.join(_config["eval_root"], eval_results_path), exist_ok=True)
    from toolkit.logger import LOGGER as logger_txt, add_log_to_file
    logger_txt.info("creating log at: {}/{}/eval_{}_{}_log.txt".format(_config["eval_root"], eval_results_path, _config["eval_task"], _config["eval_data"]))
    add_log_to_file(os.path.join(_config["eval_root"], eval_results_path, "eval_{}_{}_log.txt".format(_config["eval_task"], _config["eval_data"])))


    model = FIBERTransformerSS(_config, logger=None)
    model.setup("test")
    model.eval()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model = torch.nn.DataParallel(model).to(device)


    def infer_batch(batch):

        with torch.no_grad():
            scores = {}
            infer = model(batch, benchmark_test='itm')
            itm_logits = model.module.itm_score(infer["cls_feats"])
            scores['itm_score_logits'] = itm_logits[:, 1]
            score = torch.nn.functional.softmax(itm_logits)[:, 1]
            scores['itm_score'] = score

            infer_imag, infer_text = model(batch, benchmark_test='itc')
            image_features = infer_imag["cls_feats"]
            text_features = infer_text["cls_feats"]
            logit_scale = model.module.logit_scale.exp().mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            score = logits_per_image.diag()
            scores['itc_score'] = score

        return scores



    # image_path = '/home/wangtan/code/ms_intern/jupyter/DALLE2_demo_image/teddybear/0_1.jpg'
    # caption = 'Two teddy bears mixing sparking chemicals on a blue table with goggles.'
    eval_root_dir = _config["eval_root"] if _config["eval_root"] is not None else os.path.join('result', _config["exp_name"])
    if not os.path.exists(os.path.join(eval_root_dir, eval_results_path)):
        os.makedirs(os.path.join(eval_root_dir, eval_results_path))

    if 'my_VG' in _config["eval_data"]:
        eval_dataset = VisualGenome_My_METER(img_root=_data_info[_config["eval_data"]]["img_root"], ann_root=_data_info[_config["eval_data"]]["ann_root"], config=_config)
    elif _config["eval_data"] == 'my_youcook2':
        eval_dataset = Youcook2_My_METER(img_root=_data_info["my_youcook2"]["img_root"], ann_root=_data_info["my_youcook2"]["ann_root"], config=_config)
    elif _config["eval_data"] == 'my_gebc':
        eval_dataset = GEBC_My_METER(img_root=_data_info["my_gebc"]["img_root"], ann_root=_data_info["my_gebc"]["ann_root"], config=_config)
    elif 'kubric' in _config["eval_data"]:
        eval_dataset = Kubric_My_METER(img_root=_data_info[_config["eval_data"]]['img_root'], ann_root=_data_info[_config["eval_data"]]['ann_root'], config=_config)
    elif _config["eval_data"] == 'winoground':
        eval_dataset = Winoground_My_METER(img_root=_data_info[_config["eval_data"]]['img_root'], ann_root=_data_info[_config["eval_data"]]['ann_root'], config=_config)
    elif 'generation' in _config["eval_data"]:
        eval_dataset = Generation_My_METER(img_root=_data_info[_config["eval_data"]]['img_root'], ann_root=_data_info[_config["eval_data"]]['ann_root'], config=_config)
    elif 'valse' in _config["eval_data"]:
        eval_dataset = VALSE_My_METER(img_root=_data_info[_config["eval_data"]]['img_root'], ann_root=_data_info[_config["eval_data"]]['ann_root'], config=_config)


    gpu_cnt = torch.cuda.device_count()
    print('use {} GPUs'.format(gpu_cnt))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=int(gpu_cnt*128), num_workers=24, shuffle=False, collate_fn=eval_dataset.collate)
    meter_score_c0_i0_all_itm_logits, meter_score_c1_i0_all_itm_logits, meter_score_c0_i1_all_itm_logits, meter_score_c1_i1_all_itm_logits = [], [], [], []
    meter_score_c0_i0_all_itm, meter_score_c1_i0_all_itm, meter_score_c0_i1_all_itm, meter_score_c1_i1_all_itm = [], [], [], []
    meter_score_c0_i0_all_itc, meter_score_c1_i0_all_itc, meter_score_c0_i1_all_itc, meter_score_c1_i1_all_itc = [], [], [], []
    device = "cuda"


    # test
    for idx, (image0, image1, encode0, encode1, caption0, caption1) in enumerate(tqdm(eval_dataloader)):

        image0, image1, encode0, encode1 = image0.to(device), image1.to(device), encode0.to(device), encode1.to(device)
        batch_c0_i0 = {"text":caption0, "image":[image0], "text_ids":encode0["input_ids"], "text_labels":encode0["input_ids"], "text_masks":encode0["attention_mask"]}
        batch_c1_i0 = {"text":caption1, "image":[image0], "text_ids":encode1["input_ids"], "text_labels":encode1["input_ids"], "text_masks":encode1["attention_mask"]}
        batch_c0_i1 = {"text":caption0, "image":[image1], "text_ids":encode0["input_ids"], "text_labels":encode0["input_ids"], "text_masks":encode0["attention_mask"]}
        batch_c1_i1 = {"text":caption1, "image":[image1], "text_ids":encode1["input_ids"], "text_labels":encode1["input_ids"], "text_masks":encode1["attention_mask"]}

        meter_score_c0_i0_dict = infer_batch(batch_c0_i0)
        meter_score_c1_i0_dict = infer_batch(batch_c1_i0)
        meter_score_c0_i1_dict = infer_batch(batch_c0_i1)
        meter_score_c1_i1_dict = infer_batch(batch_c1_i1)

        meter_score_c0_i0_all_itm_logits.append(meter_score_c0_i0_dict["itm_score_logits"].cpu())
        meter_score_c1_i0_all_itm_logits.append(meter_score_c1_i0_dict["itm_score_logits"].cpu())
        meter_score_c0_i1_all_itm_logits.append(meter_score_c0_i1_dict["itm_score_logits"].cpu())
        meter_score_c1_i1_all_itm_logits.append(meter_score_c1_i1_dict["itm_score_logits"].cpu())

        meter_score_c0_i0_all_itm.append(meter_score_c0_i0_dict["itm_score"].cpu())
        meter_score_c1_i0_all_itm.append(meter_score_c1_i0_dict["itm_score"].cpu())
        meter_score_c0_i1_all_itm.append(meter_score_c0_i1_dict["itm_score"].cpu())
        meter_score_c1_i1_all_itm.append(meter_score_c1_i1_dict["itm_score"].cpu())

        meter_score_c0_i0_all_itc.append(meter_score_c0_i0_dict["itc_score"].cpu())
        meter_score_c1_i0_all_itc.append(meter_score_c1_i0_dict["itc_score"].cpu())
        meter_score_c0_i1_all_itc.append(meter_score_c0_i1_dict["itc_score"].cpu())
        meter_score_c1_i1_all_itc.append(meter_score_c1_i1_dict["itc_score"].cpu())

    meter_score_c0_i0_all_itm_logits = torch.cat(meter_score_c0_i0_all_itm_logits, dim=0)
    meter_score_c1_i0_all_itm_logits = torch.cat(meter_score_c1_i0_all_itm_logits, dim=0)
    meter_score_c0_i1_all_itm_logits = torch.cat(meter_score_c0_i1_all_itm_logits, dim=0)
    meter_score_c1_i1_all_itm_logits = torch.cat(meter_score_c1_i1_all_itm_logits, dim=0)

    meter_score_c0_i0_all_itm = torch.cat(meter_score_c0_i0_all_itm, dim=0)
    meter_score_c1_i0_all_itm = torch.cat(meter_score_c1_i0_all_itm, dim=0)
    meter_score_c0_i1_all_itm = torch.cat(meter_score_c0_i1_all_itm, dim=0)
    meter_score_c1_i1_all_itm = torch.cat(meter_score_c1_i1_all_itm, dim=0)

    meter_score_c0_i0_all_itc = torch.cat(meter_score_c0_i0_all_itc, dim=0)
    meter_score_c1_i0_all_itc = torch.cat(meter_score_c1_i0_all_itc, dim=0)
    meter_score_c0_i1_all_itc = torch.cat(meter_score_c0_i1_all_itc, dim=0)
    meter_score_c1_i1_all_itc = torch.cat(meter_score_c1_i1_all_itc, dim=0)


    eval_meter_scores_itm_logits = {'raw_info':eval_dataset.sample_pair, "c0_i0": meter_score_c0_i0_all_itm_logits, "c0_i1": meter_score_c0_i1_all_itm_logits, "c1_i0": meter_score_c1_i0_all_itm_logits, "c1_i1": meter_score_c1_i1_all_itm_logits, "model_path": _config["load_path"]}
    eval_meter_scores_itm = {'raw_info':eval_dataset.sample_pair, "c0_i0": meter_score_c0_i0_all_itm, "c0_i1": meter_score_c0_i1_all_itm, "c1_i0": meter_score_c1_i0_all_itm, "c1_i1": meter_score_c1_i1_all_itm, "model_path": _config["load_path"]}
    eval_meter_scores_itc = {'raw_info':eval_dataset.sample_pair, "c0_i0": meter_score_c0_i0_all_itc, "c0_i1": meter_score_c0_i1_all_itc, "c1_i0": meter_score_c1_i0_all_itc, "c1_i1": meter_score_c1_i1_all_itc, "model_path": _config["load_path"]}
    torch.save(eval_meter_scores_itm_logits, '{}/{}/fiber_{}_noadapter_itm_logits_scores_{}.torch'.format(eval_root_dir, eval_results_path, _config["eval_task"], _config["eval_data"]))

    # calculate the score
    def text_correct(result):
        return torch.logical_and(result["c0_i0"] > result["c1_i0"], result["c1_i1"] > result["c0_i1"])

    def image_correct(result):
        return torch.logical_and(result["c0_i0"] > result["c0_i1"], result["c1_i1"] > result["c1_i0"])

    def group_correct(result):
        return torch.logical_and(image_correct(result), text_correct(result))

    def cal_score(list_correct):
        correct_cnt = list_correct.sum()
        denominator = len(list_correct)
        return correct_cnt / denominator

    def cal_valse_score_pair(result):
        return (result["c0_i0"]>result["c1_i0"])

    def cal_valse_score_pc(result):
        return (result["c0_i0"]>0.5)

    def cal_valse_score_pf(result):
        return (result["c1_i0"]<0.5)

    def cal_valse_score_acc(result):
        true_cnt = (result["c0_i0"]>0.5).sum() + (result["c1_i0"]<0.5).sum()
        return true_cnt / (len(result["c0_i0"])*2)


    assert len(eval_meter_scores_itm["c0_i0"]) == len(text_correct(eval_meter_scores_itm))
    if 'valse' in _config["eval_data"]:
        score_valse_pair = cal_score(cal_valse_score_pair(eval_meter_scores_itm))
        score_valse_pc = cal_score(cal_valse_score_pc(eval_meter_scores_itm))
        score_valse_pf = cal_score(cal_valse_score_pf(eval_meter_scores_itm))
        score_valse_acc = cal_valse_score_acc(eval_meter_scores_itm)
        logger_txt.info("============ ITM Score =============")
        logger_txt.info("valse score pair: {}".format(score_valse_pair))
        logger_txt.info("valse score pc: {}".format(score_valse_pc))
        logger_txt.info("valse score pf: {}".format(score_valse_pf))
        logger_txt.info("valse score acc: {}".format(score_valse_acc))

    else:
        text_score_itm, image_score_itm, group_score_itm = cal_score(text_correct(eval_meter_scores_itm)), cal_score(image_correct(eval_meter_scores_itm)), cal_score(group_correct(eval_meter_scores_itm))
        eval_meter_scores_itm["scores"] = {"text_score":text_score_itm, "image_score":image_score_itm, "group_score":group_score_itm}
        torch.save(eval_meter_scores_itm, '{}/{}/fiber_{}_noadapter_itm_scores_{}.torch'.format(eval_root_dir, eval_results_path, _config["eval_task"], _config["eval_data"]))

        logger_txt.info("============ ITM Score =============")
        logger_txt.info("text score: {}".format(text_score_itm))
        logger_txt.info("image score: {}".format(image_score_itm))
        logger_txt.info("group score: {}".format(group_score_itm))


    assert len(eval_meter_scores_itc["c0_i0"]) == len(text_correct(eval_meter_scores_itc))
    if 'valse' in _config["eval_data"]:
        score_valse_pair = cal_score(cal_valse_score_pair(eval_meter_scores_itm))
        score_valse_pc = cal_score(cal_valse_score_pc(eval_meter_scores_itm))
        score_valse_pf = cal_score(cal_valse_score_pf(eval_meter_scores_itm))
        score_valse_acc = cal_valse_score_acc(eval_meter_scores_itm)
        logger_txt.info("============ ITM Score =============")
        logger_txt.info("valse score pair: {}".format(score_valse_pair))
        logger_txt.info("valse score pc: {}".format(score_valse_pc))
        logger_txt.info("valse score pf: {}".format(score_valse_pf))
        logger_txt.info("valse score acc: {}".format(score_valse_acc))

    else:
        text_score_itc, image_score_itc, group_score_itc = cal_score(text_correct(eval_meter_scores_itc)), cal_score(image_correct(eval_meter_scores_itc)), cal_score(group_correct(eval_meter_scores_itc))
        eval_meter_scores_itc["scores"] = {"text_score":text_score_itc, "image_score":image_score_itc, "group_score":group_score_itc}
        torch.save(eval_meter_scores_itc, '{}/{}/fiber_{}_noadapter_itc_scores_{}.torch'.format(eval_root_dir, eval_results_path, _config["eval_task"], _config["eval_data"]))

        logger_txt.info("============ ITC Score =============")
        logger_txt.info("text score: {}".format(text_score_itc))
        logger_txt.info("image score: {}".format(image_score_itc))
        logger_txt.info("group score: {}".format(group_score_itc))

