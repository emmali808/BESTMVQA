# MMBERT: Multimodal BERT Pretraining for Improved Medical VQA
**This model is developed based on the [model](https://github.com/VirajBagal/MMBERT) proposed in this [paper](https://arxiv.org/abs/2104.01394).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./MMBERT
conda activate mmbert
```
To fine-tune and test MMBERT on medvqa2019 dataset, please navigate to /vqamed2019 folder, else to /vqarad folder.


## medvqa2019 dataset
**Evaluate MMBERT model on medvqa2019 dataset.**
### Fine-tuning
`python train.py --run_name  hxj_test --batch_size 16 --num_vis 5 --epochs 1`

### Testing
`python test_medvqa.py --run_name hxj --num_vis 5`


## vqa-ovqa dataset
**Evaluate MMBERT model on OVQA dataset.**
### Fine-tuning
`python train_vqarad.py --run_name ovqa_test --epochs 80 --data_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --save_dir /home/coder/projects/MMBERT/vqa_ovqa --use_pretrained --batch_size 64`
### Testing
`python test_vqarad.py --run_name ovqa_test  --data_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --fine_tune_model_dir /home/coder/projects/MMBERT/vqa_ovqa/ovqa_test_acc.pt`


## vqa-rad dataset
**Evaluate MMBERT model on vqa-rad dataset.**
### Fine-tuning
`python train_vqarad.py --run_name hxj_test --epochs 1`

### Testing
`python test_vqarad.py --run_name give_name --use_pretrained`


## slake dataset
**Evaluate MMQ model on slake dataset.**
### Fine-tuning
`python train_vqarad.py --run_name hxj_test --epochs 100 --data_dir /home/coder/projects/Med-VQA/data_SLAKE --save_dir /home/coder/projects/MMBERT/vqa_slake --use_pretrained --batch_size 8`

### Testing
`python test_vqarad.py --run_name hxj_test  --data_dir /home/coder/projects/Med-VQA/data_SLAKE --fine_tune_model_dir /home/coder/projects/MMBERT/vqa_slake/hxj_test_test_acc.pt`


## PATH dataset
### Fine-tuning
**Evaluate MMBERT model on PATH dataset.**
`python train_vqarad.py --run_name hxj_test --epochs 1 --data_dir /home/coder/projects/Med-VQA/data_PATH --save_dir /home/coder/projects/MMBERT/vqa_path --use_pretrained --batch_size 8`

### Testing
`python test_vqarad.py --run_name hxj_test  --data_dir /home/coder/projects/Med-VQA/data_PATH --fine_tune_model_dir /home/coder/projects/MMBERT/vqa_path/hxj_test_test_acc.pt`