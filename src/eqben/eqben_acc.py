import torch


# calculate the score proposed in wionground
def cal_wino_score(result):
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

    return cal_score(text_correct(result)), cal_score(image_correct(result)), cal_score(group_correct(result))


# calculate the score proposed in VALSE
def cal_valse_score(result):
    def cal_valse_score_pair(result):
        return (result["c0_i0"] > result["c1_i0"])

    def cal_valse_score_pc(result):
        return (result["c0_i0"] > 0.5)

    def cal_valse_score_pf(result):
        return (result["c1_i0"] < 0.5)

    def cal_valse_score_acc(result):
        true_cnt = (result["c0_i0"] > 0.5).sum() + (result["c1_i0"] < 0.5).sum()
        return true_cnt / (len(result["c0_i0"]) * 2)

    def cal_score(list_correct):
        correct_cnt = list_correct.sum()
        denominator = len(list_correct)
        return correct_cnt / denominator

    return cal_valse_score_acc(result), min(cal_score(cal_valse_score_pc(result)), cal_score(cal_valse_score_pf(result))), cal_score(cal_valse_score_pair(result))

