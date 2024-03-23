# Cross-Modal Self-Attention with Multi-Task Pre-Training for Medical Visual Question Answering
**This model is developed based on the [model](https://github.com/haifangong/CMSA-MTPT-4-MedicalVQA) proposed in this [paper](https://www.researchgate.net/publication/351229736_Cross-Modal_Self-Attention_with_Multi-Task_Pre-Training_for_Medical_Visual_Question_Answering#fullTextFileContent).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./CMSA-MTPT-4-MedicalVQA
conda activate cmsa
```

## vqa-rad dataset
**Evaluate CMSA model on VQA-RAD dataset.**
### Fine-tuning
`python main.py --model CMSA --RAD_dir /home/coder/projects/SystemDataset/data_RAD --use_spatial  --output saved_models_data_RAD/CMSA-Bio-MTPT --epochs 60 --emb_init glove --record_id 1 --gpu 0`
### Testing
Epoch specifies which epoch model to use.
`python test.py --model CMSA --RAD_dir /home/coder/projects/SystemDataset/data_RAD --distmodal --use_spatial --input saved_models_data_RAD/CMSA-Bio-MTPT --epoch 0 --output results/ --emb_init glove`


## vqa-ovqa dataset
**Evaluate CMSA model on OVQA dataset.**
### Fine-tuning
`python main.py --model CMSA --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --use_spatial  --output saved_models_data_OVQA/CMSA-Bio-MTPT --epochs 60 --emb_init glove --record_id 1 --gpu 0`
### Testing
Epoch specifies which epoch model to use.
`python test.py --model CMSA --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --distmodal --use_spatial --input saved_models_data_OVQA/CMSA-Bio-MTPT --epoch 0 --output results/ --emb_init glove`


## PATH dataset
### Fine-tuning
**Evaluate CMSA model on PATH dataset.**
`python main.py --model CMSA --RAD_dir /home/coder/projects/Med-VQA/data_PATH --use_spatial  --output saved_models_data_PATH/CMSA-Bio-MTPT --epochs 80 --emb_init glove`

### Testing
Epoch specifies which epoch model to use.
`python test.py --model CMSA --RAD_dir /home/coder/projects/Med-VQA/data_PATH --distmodal --use_spatial --input saved_models_data_PATH/CMSA-Bio-MTPT --epoch 0 --output results/ --emb_init glove`


## slake dataset
**Evaluate MEVF model on slake dataset.**
### Fine-tuning
`python main.py --model CMSA --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --use_spatial  --output saved_models_data_SLAKE/CMSA-Bio-MTPT --epochs 60 --emb_init glove`

### Testing
Epoch specifies which epoch model to use
`python test.py --model CMSA --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --distmodal --use_spatial --input saved_models_data_SLAKE/CMSA-Bio-MTPT --epoch 79 --output results/ --emb_init glove`


## medvqa2019 dataset
**Evaluate CMSA model on medvqa2019 dataset.**
### Fine-tuning
`python main.py --model CMSA --RAD_dir /home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019 --use_spatial  --output saved_models_data_Med2019/CMSA-Bio-MTPT --epochs 80 --emb_init glove`

### Testing
`python test.py --model CMSA --RAD_dir /home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019 --distmodal --use_spatial --input saved_models_data_Med2019/CMSA-Bio-MTPT --epoch 59 --output results/ --emb_init glove`
