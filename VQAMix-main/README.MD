# VQAMix: Conditional Triplet Mixup for Medical Visual Question Answering
**This model is developed based on the [model](https://github.com/haifangong/VQAMix) proposed in this [paper](https://www.researchgate.net/publication/361364110_VQAMix_Conditional_Triplet_Mixup_for_Medical_Visual_Question_Answering).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./VQAMix-main
conda activate VQAMix
```


## OVQA dataset
### BAN
### Fine-tuning
`python main.py --model BAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --output saved_model_OVQA/BAN_MEVF --lr 0.03 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3 --batch_size 64`

### Testing
`python test.py --model BAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --input saved_model_OVQA/BAN_MEVF/ --output="results-BAN-OVQA" --model_name model_epoch_best.pth`

### SAN
### Fine-tuning
`python main.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --output saved_model_OVQA/SAN_MEVF --lr 0.03 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3 --batch_size 64`
### Testing
python test.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --input saved_model_OVQA/SAN_MEVF/ --output="results-SAN-OVQA" --model_name model_epoch_best.pth

## SLAKE dataset
### Fine-tuning
`python main.py --model BAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --output saved_model_SLAKE/BAN_MEVF --lr 0.03 --epochs 1 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3`

### Testing
`python test.py --model BAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --input saved_model_SLAKE/BAN_MEVF/ --output="results-SLAKE"`

## Med2019 dataset
### Fine-tuning
`python main.py --model BAN --use_RAD --RAD_dir /home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019 --output saved_model_Med2019/BAN_MEVF --lr 0.03 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3`

### Testing
`python test.py --model BAN --use_RAD --RAD_dir /home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019 --input saved_model_Med2019/BAN_MEVF --output="results-Med2019"`