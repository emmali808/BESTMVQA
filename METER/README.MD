# An Empirical Study of Training End-to-End Vision-and-Language Transformers
**This model is developed based on the [model](https://github.com/zdou0830/METER) proposed in this [paper](https://arxiv.org/abs/2111.02387).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./METER
conda activate meter02
```
Before fine-tuning and testing, you need to generate the arrow files for each dataset. Please refer to **generate_arrow.ipynb**

## vqa-rad dataset
**Evaluate METER model on VQA-RAD dataset.**
### Fine-tuning
`python run.py with data_root=/home/coder/projects/METER/data/vqa_ovqa num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=16 load_path=meter_clip16_288_roberta_pretrain.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=15 vocab_size=10000`
### Testing
You need to modify the loading path of your model.
python run.py with data_root=/home/coder/projects/METER/data/vqa_rad num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=2 load_path=/home/coder/projects/METER/result/finetune_vqa_seed0_from_meter_clip16_288_roberta_pretrain/version_45/checkpoints/last.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=1 vocab_size=10000 test_only=True


## vqa-ovqa dataset
**Evaluate METER model on OVQA dataset.**
### Fine-tuning
`python run.py with data_root=/home/coder/projects/METER/data/vqa_ovqa_v2.0 num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=16  task_vqa_ovqa load_path=meter_clip16_288_roberta_pretrain.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=10 vocab_size=10000`
### Testing
You need to modify the loading path of your model.
`python run.py with data_root=/home/coder/projects/METER/data/vqa_ovqa num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=2 task_vqa_ovqa load_path=/home/coder/projects/METER/result/finetune_vqa_seed0_from_meter_clip16_288_roberta_pretrain/version_45/checkpoints/last.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=1 vocab_size=10000 test_only=True`


## PATH dataset
### Fine-tuning
**Evaluate METER model on PATH dataset.**
`python run.py with data_root=/home/coder/projects/METER/data/vqa_path num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=4 load_path=meter_clip16_288_roberta_pretrain.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=15 vocab_size=10000 task_vqa_path`

### Testing
You need to modify the loading path of your model.
`python run.py with data_root=/home/coder/projects/METER/data/vqa_path num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=2 load_path=/home/coder/projects/METER/result/task_vqa_path_seed0_from_meter_clip16_288_roberta_pretrain/version_1/checkpoints/last.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=10 vocab_size=10000 test_only=True task_vqa_path`


## slake dataset
**Evaluate METER model on slake dataset.**
### Fine-tuning
`python run.py with data_root=/home/coder/projects/METER/data/vqa_slake num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=8 load_path=meter_clip16_288_roberta_pretrain.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=15 vocab_size=10000`

### Testing
You need to modify the loading path of your model.
`python run.py with data_root=/home/coder/projects/METER/data/vqa_slake num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=2 load_path=meter_clip16_288_roberta_pretrain.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=1 vocab_size=10000 test_only=True`


## medvqa2019 dataset
**Evaluate METER model on medvqa2019 dataset.**
### Fine-tuning
`python run.py with data_root=/home/coder/projects/METER/data/vqa_slake num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=8 load_path=meter_clip16_288_roberta_pretrain.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=15 vocab_size=10000`

### Testing
You need to modify the loading path of your model.
`python run.py with data_root=/home/coder/projects/METER/data/vqa_slake num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=2 load_path=meter_clip16_288_roberta_pretrain.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=1 vocab_size=10000 test_only=True`

