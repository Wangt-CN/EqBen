import os
import shutil

root_path = '/datadrive_d/wangtan/azure_storage/vigstandard_data/v-tanwang/code/remedy/fiber/results_pairloss_hard_itm'
inter_path = 'version_0/checkpoints'

dir_exp_list = os.listdir(root_path)
for exp_name in dir_exp_list:
    model_dir_path = os.path.join(root_path, exp_name, inter_path)
    model_file_name_list = os.listdir(model_dir_path)
    if 'val-best.ckpt' in model_file_name_list:
        # has the model file
        print('has the model, thus skip this exp dir')
        continue
    else:
        assert len(model_file_name_list) == 2
        for model_file_name in model_file_name_list:
            if model_file_name != 'last.ckpt':
                base_model_name = model_file_name
        base_model_path = os.path.join(root_path, exp_name, inter_path, base_model_name)
        new_model_path = os.path.join(root_path, exp_name, inter_path, 'val-best.ckpt')
        shutil.copy(base_model_path, new_model_path)
        print('copy to {}'.format(new_model_path))