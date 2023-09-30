### EqSim - FIBER

#### Env Setup
```bash
conda create -n eqsim python=3.8
conda activate eqsim
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r ./requirements.txt
pip install torchmetrics==0.5
pip install -e .
```


#### Preparation
- Dataset: We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details. You will mainly need Flickr30K data.
- FIBER Checkpoint: Please download [FIBER checkpoint](https://storage.googleapis.com/eqben-data/fiber_pretrain.ckpt) here and put it to the path you want.


#### Training
```python
python run.py with data_root=/path/to/pyarrow/dir \
num_gpus=8 num_nodes=1 task_finetune_irtr_itm_f30k_hardneg per_gpu_batchsize=8 load_path=/path/to/fiber/checkpoint.pth \
log_dir=/path/to/log/dir \
exp_name=task_finetune_irtr_itm_hardneg_f30k_ftsize288_ours \
image_size=288 learning_rate=2e-5 \
pairloss='hard_easy_itc' pairloss_text_side=True pairloss_img_side=True pairloss_pos_reg=False pairloss_weight=0.5 pairloss_norm=False pairloss_margin=0.1 \
sample_num=4 score_activation=None score_pre_softmax=True
```


#### Evaluation
You can follow the method [here](https://github.com/Wangt-CN/EqBen#eqben-1) to perform the evaluation.