# BESTMVQA: A Benchmark Evaluation System for Medical Visual Question Answering

**Tips: The complete  code will be published after the paper is included.**

[Paper]

This is the repository for the ASMVQA, a unified assistant system for users to conveniently perform diverse medical practice. The demonstration video of our system can be found at https://youtu.be/QkEeFlu1x4A. The repository contains:

- Overview
- Med-VQA Data for Pre-training and Fine-tuning
- Install

## Overview

Medical Visual Question Answering (Med-VQA) is a task that answers a natural language question with a medical image. Existing VQA techniques can be directly applied to solving the task. However, they often suffer from (i) the data insufficient problem, which makes it difficult to train the state of the arts (SOTAs) for domain-specific tasks, and (ii) the reproducibility problem, that existing models have not been thoroughly evaluated in a unified experimental setup. To address the issues, we develop a Benchmark Evaluation SysTem for Medical Visual Question Answering, denoted by BESTMVQA. Given clinical data, our system provides a useful tool for users to automatically build Med-VQA datasets. Users can conveniently select a wide spectrum of models from our library to perform a comprehensive evaluation study. With simple configurations, our system can automatically train and evaluate the selected models over a benchmark dataset, and reports the comprehensive results for users to develop new techniques or perform medical practice. Limitations of existing work are overcome (i) by the data generation tool, which automatically constructs new datasets from unstructured clinical data, and (ii) by evaluating SOTAs on benchmark datasets in a unified experimental setup. The demonstration video of our system can be found at https://youtu.be/QkEeFlu1x4A.

<img src="/overview.png" alt="Overview" style="zoom:20%;" />

## Med-VQA Data for Pre-training and Fine-tuning

We provide the benchmark Med-VQA datasets for practice, you can download them from Google Cloud.

## Install

1. Clone this repository and navigate to Demo/VQASystem:

2. Install Package:  Please refer to the envs_files folder to create conda environments of the models we provide. 

3. We use MySQL for data storage,  you can restore the database by refering to db.sql:

```sql
sudo mysql
create database vqa;
use vqa;
source /your_path/db.sql;
```

4. Configure model config: Please deploy the download datasets and modify the model config in file /Demo/VQASystem/model_info.py. An example of modifying the config:

```
tbh
```

5. Run Application: We use streamlit framework to develop our App.

```
pip install streamlit
streamlit run Home.py
```

If you're interested in some of the model provided in our model library, you can check the readme file in relevant model folder.







