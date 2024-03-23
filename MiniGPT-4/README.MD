# MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models
**This model is developed based on the [model](https://minigpt-4.github.io) proposed in this [paper](https://arxiv.org/pdf/2310.09478.pdf).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./MiniGPT-4
conda activate minigpt4
```

### Single data test instruction
`python demo_hxj02.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 1`

### Dataset evaluation results
No need to switch environments

`python evaluate.py --data_dir vqa-rad --record_id 1`