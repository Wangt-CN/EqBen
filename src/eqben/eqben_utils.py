import json
import os
import torch
from torch.utils.data import Dataset


############# EqBen ############
class EqBenALL(Dataset):
    def __init__(self, img_root, ann_root, config, customized_data_toolkit):
        super(EqBenALL, self).__init__()
        self.config = config
        self.img_root = img_root
        self.global_ann = json.load(open(ann_root, 'r'))
        self.customized_data_toolkit = customized_data_toolkit

    def __getitem__(self, index):
        database = self.global_ann[index]['image'].split('/')[0]
        img_path = os.path.join(self.img_root, self.global_ann[index]['image'])
        caption = self.global_ann[index]['caption']
        if database == 'eqbenyoucook2':
            image = self.customized_data_toolkit.process_img_npy(img_path)
        else:
            image = self.customized_data_toolkit.process_img_pixel(img_path)
        return image, caption

    def __len__(self):
        return len(self.global_ann)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, 'collate'): # use customized collact
            return self.customized_data_toolkit.collate_eqben(batch)
        else: # use the default pytorch collact
            return torch.utils.data.default_collate(batch)




class EqBenAG(Dataset):
    def __init__(self, img_root, ann_root, config, customized_data_toolkit):
        super(EqBenAG, self).__init__()
        self.config = config
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(self.global_ann.items()):
            print('Loading data {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            for subject_dir, subject_dir_values in video_dir_values.items():
                for frame_info in subject_dir_values:
                    image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
                    caption0, caption1 = frame_info['caption0'][0], frame_info['caption1'][0] # only use the first caption in the caption list
                    added_sample = {'subject':subject_dir, 'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
                    self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = self.customized_data_toolkit.process_img_pixel(image0), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, 'collate'): # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else: # use the default pytorch collact
            return torch.utils.data.default_collate(batch)



class EqBenYoucook2(Dataset):
    def __init__(self, img_root, ann_root, config, customized_data_toolkit):
        super(EqBenYoucook2, self).__init__()
        self.config = config
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(self.global_ann.items()):
            print('Loading data {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            for frame_info in video_dir_values:
                image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
                caption0, caption1 = frame_info["caption0"], frame_info["caption1"]
                added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
                self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = self.customized_data_toolkit.process_img_npy(image0), self.customized_data_toolkit.process_img_npy(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, 'collate'): # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else: # use the default pytorch collact
            return torch.utils.data.default_collate(batch)



class EqBenGEBC(Dataset):
    def __init__(self, img_root, ann_root, config, customized_data_toolkit):
        super(EqBenGEBC, self).__init__()
        self.config = config
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(self.global_ann.items()):
            print('Loading data {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            for frame_info in video_dir_values:
                image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
                caption0, caption1 = frame_info["caption0"], frame_info["caption1"]
                added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
                self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = self.customized_data_toolkit.process_img_pixel(image0), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, 'collate'): # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else: # use the default pytorch collact
            return torch.utils.data.default_collate(batch)




class EqBenKubric(Dataset):
    def __init__(self, img_root, ann_root, config, customized_data_toolkit):
        super(EqBenKubric, self).__init__()
        self.config = config
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, frame_info in enumerate(self.global_ann):
            print('Loading data {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
            caption0, caption1 = frame_info['caption0'], frame_info['caption1']
            added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
            self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = self.customized_data_toolkit.process_img_pixel(image0), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, 'collate'): # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else: # use the default pytorch collact
            return torch.utils.data.default_collate(batch)




class EqBenSD(Dataset):
    def __init__(self, img_root, ann_root, config, customized_data_toolkit):
        super(EqBenSD, self).__init__()
        self.config = config
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        self.customized_data_toolkit = customized_data_toolkit

        for process_idx, frame_info in enumerate(self.global_ann):
            print('Loading data {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            image0_path, image1_path = os.path.join(img_root, frame_info['image0']), os.path.join(img_root, frame_info['image1'])
            caption0, caption1 = frame_info['caption0'], frame_info['caption1']
            added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
            self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = self.customized_data_toolkit.process_img_pixel(image0), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, 'collate'): # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else: # use the default pytorch collact
            return torch.utils.data.default_collate(batch)



####### Conventional Benchmarks #######

class Winoground(Dataset):
    def __init__(self, img_root, ann_root, config, customized_data_toolkit):
        super(Winoground, self).__init__()
        self.config = config
        self.sample_pair = []
        self.global_ann = [json.loads(example_json) for example_json in open(ann_root).readlines()]
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, frame_info in enumerate(self.global_ann):
            print('Loading data {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            image0_path, image1_path = os.path.join(img_root, frame_info['image_0']+'.png'), os.path.join(img_root, frame_info['image_1']+'.png')
            caption0, caption1 = frame_info['caption_0'], frame_info['caption_1']
            added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
            self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = self.customized_data_toolkit.process_img_pixel(image0), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, 'collate'): # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else: # use the default pytorch collact
            return torch.utils.data.default_collate(batch)




class VALSE(Dataset):
    def __init__(self, img_root, ann_root, config, customized_data_toolkit):
        super(VALSE, self).__init__()
        self.config = config
        self.sample_pair = []
        self.global_ann = json.load(open(ann_root, 'r'))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, frame_info in enumerate(self.global_ann):
            print('Loading data {}/{} '.format(process_idx, len(self.global_ann)), end='\r')
            image0_path, image1_path = os.path.join(img_root, frame_info['image_0']), os.path.join(img_root, frame_info['image_1'])
            caption0, caption1 = frame_info['caption_0'], frame_info['caption_1']
            added_sample = {'image0':image0_path, 'caption0':caption0, 'image1':image1_path, 'caption1':caption1}
            self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = self.sample_pair[index]['image0'], self.sample_pair[index]['caption0'], \
                                                      self.sample_pair[index]['image1'], self.sample_pair[index]['caption1']
        image0, image1 = self.customized_data_toolkit.process_img_pixel(image0), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, 'collate'): # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else: # use the default pytorch collact
            return torch.utils.data.default_collate(batch)
