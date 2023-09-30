import os
import sys
import json
import numpy as np

SUBSET_LENGTH_INFO = {
    'eqbenag': 195872,
    'eqbenyoucook2': 45849,
    'eqbengebc': 1814,
    'eqbenkubric_loc': 2000,
    'eqbenkubric_attr': 2000,
    'eqbenkubric_cnt': 2000,
    'eqbensd': 1513,
}


private_idx = json.load(open("src_file/private_random_idx.json", 'r'))
private_info = json.load(open("src_file/private_info.json", 'r'))

try:
    submission_score_file = os.path.join("src_file/prediction.json")
    submission_score = json.load(open(submission_score_file, 'r'))
except:
    submission_score = np.load("src_file/prediction.npy")

recover_idx = np.argsort(private_idx)
score_list_recover = submission_score[recover_idx]  # re-sort

#### divide score list to different dataset
class eqben_score():
    def __init__(self):
        sample_length = len(submission_score) // 4
        self.score_list_dataset = {'c0i0': np.zeros(sample_length), 'c0i1': np.zeros(sample_length), 'c1i0': np.zeros(sample_length), 'c1i1': np.zeros(sample_length)}

    def update(self, idx, score, info):
        dataset = info['dataset']
        item_name = info['name']
        self.score_list_dataset[item_name][info['sample_cnt']] = score

        if not hasattr(self, dataset):
            setattr(self, dataset, {'start': idx, 'end': idx + 1})
        else:
            temp = getattr(self, dataset)
            temp['end'] += 1
            setattr(self, dataset, temp)


# process the score for each dataset
eqben_score_dict = eqben_score()
for idx, (every_score, every_info) in enumerate(zip(score_list_recover, private_info)):
    eqben_score_dict.update(idx, every_score, every_info)


# calculate the score proposed in wionground
def cal_wino_score(result):
    def text_correct(result):
        return np.logical_and(result["c0i0"] > result["c1i0"], result["c1i1"] > result["c0i1"])

    def image_correct(result):
        return np.logical_and(result["c0i0"] > result["c0i1"], result["c1i1"] > result["c1i0"])

    def group_correct(result):
        return np.logical_and(image_correct(result), text_correct(result))

    def cal_score(list_correct):
        correct_cnt = list_correct.sum()
        denominator = len(list_correct)
        return correct_cnt / denominator

    return cal_score(text_correct(result)), cal_score(image_correct(result)), cal_score(group_correct(result))


scores = {}
for dataset in SUBSET_LENGTH_INFO.keys():
    dataset_score_idx = getattr(eqben_score_dict, dataset)
    idx_start, idx_end = dataset_score_idx['start'] // 4, dataset_score_idx['end'] // 4
    dataset_score = {j: k[idx_start:idx_end] for j, k in eqben_score_dict.score_list_dataset.items()}
    text_score, image_score, group_score = cal_wino_score(dataset_score)
    scores[dataset] = {'text': text_score, 'image': image_score, 'group': group_score}

print(scores)
