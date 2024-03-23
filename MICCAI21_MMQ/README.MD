# Multiple Meta-model Quantifying for Medical Visual Question Answering (MMQ-VQA)
**This model is developed based on the [model](https://github.com/aioz-ai/MICCAI21_MMQ) proposed in this [paper](https://arxiv.org/abs/2105.08913).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./MICCAI21_MMQ
conda activate mmq
```

## vqa-ovqa dataset
**Evaluate MMQ model on OVQA dataset.**
### Fine-tuning
`python3 main.py --use_VQA --VQA_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml.weights --maml_nums 0 --model BAN --lr 0.001 --seed 1342 --output saved_models/MMQ_BAN_MEVF_OVQA --epoch 60`
### Testing
`python test.py --use_VQA --VQA_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml.weights --input saved_models/MMQ_BAN_MEVF_OVQA --maml_nums 0 --model BAN --epoch _last`


## vqa-rad dataset
**Evaluate MMQ model on vqa-rad dataset.**
### Fine-tuning
`python3 main.py --use_VQA --VQA_dir /home/coder/projects/SystemDataset/data_RAD --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml.weights --maml_nums 0 --model BAN --lr 0.001 --seed 1342 --output saved_models/MMQ_BAN_MEVF_RAD --epoch 60`

### Testing
`python test.py --use_VQA --VQA_dir /home/coder/projects/SystemDataset/data_RAD --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml.weights --input saved_models/MMQ_BAN_MEVF_RAD --maml_nums 0 --model BAN --epoch _last`


## medvqa2019 dataset
**Evaluate MMQ model on medvqa2019 dataset.**
### Fine-tuning
`python3 main.py --use_VQA --VQA_dir /home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019 --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml.weights --maml_nums 0 --model BAN --lr 0.001 --seed 1342 --output saved_models/MMQ_BAN_MEVF_Med2019  --epochs 1`

### Testing
`python test.py --use_VQA --VQA_dir /home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019 --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml.weights --input saved_models/MMQ_BAN_MEVF_Med2019 --maml_nums 0 --model BAN --epoch _last`


## slake dataset
**Evaluate MMQ model on slake dataset.**
### Fine-tuning
`python main.py --use_VQA --VQA_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --feat_dim 32 --img_size 84 --maml_model_path pretrained_maml.weights --maml_nums 0 --model BAN --lr 0.01 --seed 2104 --output saved_models/MMQ_BAN_MEVF_SLAKE --epochs 1`

### Testing
`python test.py --use_VQA --VQA_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --feat_dim 32 --img_size 84 --maml_model_path pretrained_maml.weights --input saved_models/MMQ_BAN_MEVF_SLAKE --maml_nums 0 --model BAN --epoch _last`


## PATH dataset
### Fine-tuning
**Evaluate MMQ model on PATH dataset.**
`python main.py --use_VQA --VQA_dir /home/coder/projects/Med-VQA/data_PATH --maml --autoencoder --feat_dim 32 --img_size 84 --maml_model_path pretrained_maml.weights --maml_nums 0 --model BAN --lr 0.01 --seed 2104 --output saved_models/MMQ_BAN_MEVF_PATH --epochs 1`

### Testing
`python test.py --use_VQA --VQA_dir /home/coder/projects/Med-VQA/data_PATH --maml --autoencoder --feat_dim 32 --img_size 84 --maml_model_path pretrained_maml.weights --input saved_models/MMQ_BAN_MEVF_PATH --maml_nums 0 --model BAN --epoch _last`