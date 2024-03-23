# Vision-Language Pre-Training with Triple Contrastive Learning, CVPR 2022
**This model is developed based on the [model](https://github.com/uta-smile/TCL) proposed in this [paper](https://arxiv.org/abs/2202.10401).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./TCL
conda activate tcl
```

## vqa-ovqa dataset
### Fine-tuning&Testing 
`CUDA_VISIBLE_DEVICES=1 nohup python VQA.py --config ./configs/VQA_ovqa.yaml --output_dir output/vqa_ovqa --checkpoint "/home/coder/projects/TCL/TCL_4M.pth"`

### Only Testing
`python VQA.py --config ./configs/VQA_ovqa.yaml --output_dir output/VQA_ovqa --checkpoint /home/coder/projects/TCL/output/vqa_ovqa/checkpoint_latest.pth --evaluate`

## vqa-rad dataset
### Fine-tuning&Testing 
`python VQA.py --config ./configs/VQA_rad.yaml --output_dir output/vqa_rad --checkpoint "/home/coder/projects/TCL/TCL_4M.pth"`

### Only Testing
`python VQA.py --config ./configs/VQA_rad.yaml --output_dir output/vqa_rad --checkpoint /home/coder/projects/TCL/output/vqa_rad/checkpoint_latest.pth --evaluate`

## vqa-path dataset
### Fine-tuning&Testing 

`python VQA.py --config ./configs/VQA_path.yaml --output_dir output/vqa_path --checkpoint "/home/coder/projects/TCL/TCL_4M.pth"`

### Only Testing

`python VQA.py --config ./configs/VQA_path.yaml --output_dir output/vqa_path --checkpoint /home/coder/projects/TCL/output/vqa_path/checkpoint_latest.pth --evaluate`

## vqa-slake dataset
### Fine-tuning&Testing 
`python VQA.py --config ./configs/VQA_slake.yaml --output_dir output/vqa_slake --checkpoint "/home/coder/projects/TCL/TCL_4M.pth"`

### Only Testing
`python VQA.py --config ./configs/VQA_slake.yaml --output_dir output/vqa_slake --checkpoint /home/coder/projects/TCL/output/vqa_slake/checkpoint_latest.pth --evaluate`

## vqa-med dataset
### Fine-tuning&Testing 
`python VQA.py --config ./configs/VQA_med.yaml --output_dir output/vqa_med --checkpoint "/home/coder/projects/TCL/TCL_4M.pth"`

### Only Testing
`python VQA.py --config ./configs/VQA_med.yaml --output_dir output/vqa_med --checkpoint /home/coder/projects/TCL/output/vqa_med/checkpoint_latest.pth --evaluate`