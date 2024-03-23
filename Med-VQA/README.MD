# Medical Visual Question Answering via Conditional Reasoning
**This model is developed based on the [model](https://github.com/Awenbocc/Med-VQA) proposed in this [paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413761).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./Med-VQA
conda activate CR
```


## vqa-ovqa dataset
### Pre-training
The generated model is in the log

`python type_classifier.py --autoencoder --maml --dataset OVQA`

### Fine-tuning&Testing
`python main.py --gpu 0 --seed 88 --data_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --record_id 1 --epochs 20`

## slake dataset
### Pre-training
The generated model is in the log

`python type_classifier.py --autoencoder --maml --dataset SLAKE`

### Fine-tuning&Testing
`python main03.py --autoencoder --maml --dataset SLAKE --data_dir /home/coder/projects/Med-VQA/data_SLAKE`

## path dataset
### Pre-training
The generated model is in the log

`python type_classifier.py --autoencoder --maml --dataset PATH`

### Fine-tuning&Testing
`python main03.py --autoencoder --maml --dataset PATH --data_dir /home/coder/projects/Med-VQA/data_PATH`

## med-2019 dataset
### Pre-training
The generated model is in the log

`python type_classifier.py --autoencoder --maml --dataset Med-2019`

### Fine-tuning&Testing
`python main03.py --autoencoder --maml --dataset Med-2019 --data_dir /home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019`

