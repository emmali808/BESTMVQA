# Multiple Meta-model Quantifying for Medical Visual Question Answering (MMQ-VQA)
**This model is developed based on the [model](https://github.com/aioz-ai/MICCAI21_MMQ) proposed in this [paper](https://arxiv.org/abs/2105.08913).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./MEVF/MICCAI19-MedVQA
conda activate MEVF
```


## vqa-ovqa dataset
**Evaluate MEVF model with SAN on OVQA dataset.**
### Fine-tuning
`python main.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --output saved_models/SAN_MEVF_OVQA --epochs 20 --lr 0.001 --batch_size 64`
### Testing
`python test.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --input saved_models/SAN_MEVF_OVQA --epoch 19 --output results/SAN_MEVF`

**Evaluate MEVF model with BAN on OVQA dataset.**
### Fine-tuning
`python main.py --model BAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --output saved_models/BAN_MEVF_OVQA --epochs 25 --lr 0.001 --batch_size 128`
### Testing
`python test.py --model BAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --input saved_models/BAN_MEVF_OVQA --epoch 24 --output results/BAN_MEVF`

## vqa-rad dataset
**Evaluate MEVF model with SAN on vqa-rad dataset.**
### Fine-tuning
`python main.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_RAD --maml --autoencoder --output saved_models/SAN_MEVF --epochs 20 --lr 0.001 --batch_size 32`

### Testing
`python test.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_RAD --maml --autoencoder --input saved_models/SAN_MEVF --epoch 19 --output results/SAN_MEVF`

## medvqa2019 dataset
**Evaluate MEVF model with BAN on medvqa2019 dataset.**
### Fine-tuning
 [CUDA_VISIBLE_DEVICES=1]
`python main02.py --model BAN --use_RAD --medvqa2019_dir data_Med/VQA-Med-2019/ --maml --autoencoder --output saved_models/BAN_MEVF_medvqa2019 --epochs 1`

### Testing
`python test_medvqa.py --model BAN --use_RAD --medvqa2019_dir data_Med/VQA-Med-2019/ --maml --autoencoder --input saved_models/BAN_MEVF_medvqa2019 --epoch 0 --output results/BAN_MEVF_medvqa2019`


## slake dataset
**Evaluate MEVF model with SAN on slake dataset.**
### Fine-tuning
Epochs must be set to be greater than 15 to save the model, or this setting can be modified by adjusting the code.

`python main.py --model SAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --output saved_models/SAN_MEVF_SLAKE --epochs 40`

### Testing
Epoch specifies which epoch model to use

`python test.py --model SAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --input saved_models/SAN_MEVF_SLAKE --epoch 15 --output saved_models/SAN_MEVF_SLAKE --record_id 1`

**Evaluate MEVF model with BAN on slake dataset.**
### Fine-tuning
Epochs must be set to be greater than 15 to save the model, or this setting can be modified by adjusting the code
`python main.py --model BAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --output saved_models/BAN_MEVF_SLAKE --epochs 1`

### Testing
Epoch specifies which epoch model to use.
`python test.py --model BAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --input saved_models/BAN_MEVF_SLAKE --epoch 0 --output saved_models/BAN_MEVF_SLAKE`


## PATH dataset
### Fine-tuning
The dataset for this baseline is stored in/dev/shm/data-PATH because the data preprocessing methods are different.

**Evaluate MEVF model with SAN on PATH dataset.**
`python main.py --model SAN --use_RAD --RAD_dir /dev/shm/data_PATH --maml --autoencoder --output saved_models/SAN_MEVF_PATH --epochs 30`

### Testing
Epoch specifies which epoch model to use.

`python test.py --model SAN --use_RAD --RAD_dir /dev/shm/data_PATH --maml --autoencoder --input saved_models/SAN_MEVF_PATH --epoch 19 --output saved_models/SAN_MEVF_PATH --record_id 1`


**Evaluate MEVF model with BAN on PATH dataset.**
### Fine-tuning
Epochs must be set to be greater than 15 to save the model, or this setting can be modified by adjusting the code.

`python main.py --model BAN --use_RAD --RAD_dir /dev/shm/data_PATH --maml --autoencoder --output saved_models/BAN_MEVF_PATH --epochs 20`

### Testing
Epoch specifies which epoch model to use.

`python test.py --model BAN --use_RAD --RAD_dir /dev/shm/data_PATH --maml --autoencoder --input saved_models/BAN_MEVF_PATH --epoch 19 --output saved_models/BAN_MEVF_PATH`
