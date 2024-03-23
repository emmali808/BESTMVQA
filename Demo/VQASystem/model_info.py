MEVF={
    'url':"https://arxiv.org/pdf/1909.11867.pdf",
    'title':"MEVF——Overcoming Data Limitation in Medical Visual Question Answering",
    'attention':["BAN","SAN"],
    "RNN":["LSTM","GRU"],
    "path":{
        "VQA-RAD":"/home/coder/projects/MEVF/MICCAI19-MedVQA/run_vqarad.sh",
        "SLAKE":"/home/coder/projects/MEVF/MICCAI19-MedVQA/run_slake.sh",
        "MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/run_medvqa.sh",
        "PATHVQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/run_path.sh"
        },
    "dataset":{"VQA-RAD":"/home/coder/projects/SystemDataset/data_RAD","MED-VQA":"data_Med/VQA-Med-2019/","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE", "PATHVQA":"/dev/shm/data_PATH"}
}

VQAMix={
    'url':"https://ieeexplore.ieee.org/abstract/document/9802503",
    'title':"VQAMix: Conditional Triplet Mixup for Medical Visual Question Answering",
    'attention':["BAN","SAN"],
    "RNN":["LSTM","GRU"],
    "path":{
        "VQA-RAD":"/home/coder/projects/VQAMix-main/run.sh",
        "PATHVQA":"/home/coder/projects/VQAMix-main/run.sh",
         "SLAKE":"/home/coder/projects/VQAMix-main/run_slake.sh",
        "MED-VQA":"/home/coder/projects/VQAMix-main/run_medvqa.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD", "PATHVQA":"data_PathVQA","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE","MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019"}
}

CR={
    'url':"https://dl.acm.org/doi/abs/10.1145/3394171.3413761",
    'title':"CR: Medical Visual Question Answering via Conditional Reasoning",
    'pretrain_dataset':{"SLAKE":"SLAKE","PATH":"PATH","Med2019":"Med-2019"},
    'pretrain_path':{"SLAKE":"/home/coder/projects/Med-VQA/pretrain.sh","PATH":"/home/coder/projects/Med-VQA/pretrain.sh","Med2019":"/home/coder/projects/Med-VQA/pretrain.sh"},
    'attention':["BAN","SAN"],
    "RNN":["LSTM","GRU"],
    "path":{
        "VQA-RAD":"/home/coder/projects/Med-VQA/run_vqarad.sh",
        "PATHVQA":"/home/coder/projects/Med-VQA/run_path.sh",
         "SLAKE":"/home/coder/projects/Med-VQA/run_slake.sh",
        "MED-VQA":"/home/coder/projects/Med-VQA/run_medvqa.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD", "PATHVQA":"data_PathVQA","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE","MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019"}
}

MMBERT={
    'url':"https://ieeexplore.ieee.org/abstract/document/9434063",
    'title':"MMBERT: Multimodal BERT Pretraining for Improved Medical VQA",
    'pretrain_dataset':{"ROCO":"/home/coder/projects/PubMedCLIP-main/PubMedCLIP/roco/ROCO/roco/roco-dataset-master/data",
                        "Mimic-CXR":"/home/coder/projects/PubMedCLIP-main/PubMedCLIP/roco/ROCO/roco/roco-dataset-master/data",
                        "MedICaT":"/home/coder/projects/PubMedCLIP-main/PubMedCLIP/roco/ROCO/roco/roco-dataset-master/data"},
    'pretrain_path':{"ROCO":"/home/coder/projects/MMBERT/pretrain.sh","Mimic-CXR":"/home/coder/projects/MMBERT/pretrain.sh",
                    "MedICaT":"/home/coder/projects/MMBERT/pretrain.sh"},
    "path":{
        "VQA-RAD":"/home/coder/projects/MMBERT/vqarad/run_vqarad.sh",
        "SLAKE":"/home/coder/projects/MMBERT/vqarad/run_slake.sh",
        "MED-VQA":"/home/coder/projects/MMBERT/vqamed2019/run_medvqa.sh",
        "PATHVQA":"/home/coder/projects/MMBERT/vqarad/run_path.sh"
        },
    "dataset":{"VQA-RAD":"/home/coder/projects/Med-VQA/data","MED-VQA":"/home/coder/projects/MMBERT/VQA-Med-2019","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE", "PATHVQA":"data_PathVQA"}
}
MMQ={
    'url':"https://arxiv.org/pdf/2105.08913.pdf",
    'title':"Multiple Meta-model Quantifying for Medical Visual Question Answering",
    'attention':["BAN","SAN"],
    "RNN":["LSTM","GRU"],
    "path":{
        "VQA-RAD":"/home/coder/projects/MICCAI21_MMQ/run_rad.sh",
        "SLAKE":"/home/coder/projects/MICCAI21_MMQ/run_slake.sh",
        "MED-VQA":"/home/coder/projects/MICCAI21_MMQ/run_med.sh",
        "PATHVQA":"/home/coder/projects/MICCAI21_MMQ/run_path.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD","MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE", "PATHVQA":"data_PathVQA"}
}

CSMA={
    'url':"https://www.researchgate.net/publication/351229736_Cross-Modal_Self-Attention_with_Multi-Task_Pre-Training_for_Medical_Visual_Question_Answering#fullTextFileContent",
    'title':"Cross-Modal Self-Attention with Multi-Task Pre-Training for Medical Visual Question Answering",
    'attention':["BAN","SAN","CMSA"],
    "RNN":["LSTM","GRU"],
    "path":{
        "VQA-RAD":"/home/coder/projects/CMSA-MTPT-4-MedicalVQA/run.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD"}
}

PTUnifier={
    'url':"https://arxiv.org/abs/2302.08958",
    'title':"PTUnifier: Towards unifying medical vision-and-language pre-training via soft prompts.",
    "path":{
        "VQA-RAD":"/home/coder/projects/PTUnifier-share/run_scripts/finetune_rad.sh",
        "PATHVQA":"/home/coder/projects/PTUnifier-share/run_scripts/finetune_path.sh",
         "SLAKE":"/home/coder/projects/PTUnifier-share/run_scripts/finetune_slake.sh",
        "MED-VQA":"/home/coder/projects/PTUnifier-share/run_scripts/finetune_med2019.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD", "PATHVQA":"data_PathVQA","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE","MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019"}
}

METER={
    'url':"http://openaccess.thecvf.com/content_cvpr_2016/html/Yang_Stacked_Attention_Networks_CVPR_2016_paper.html",
    'title':"METER：Stacked Attention Networks for Image Question Answering",
    "path":{
        "VQA-RAD":"/home/coder/projects/METER/run_meter_rad.sh",
        "PATHVQA":"/home/coder/projects/METER/run_meter_path.sh",
         "SLAKE":"/home/coder/projects/METER/run_meter_slake.sh",
        "MED-VQA":"/home/coder/projects/METER/run_meter_med.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD", "PATHVQA":"data_PathVQA","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE","MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019"}
}

MedVInT={
    'url':"https://arxiv.org/abs/2305.10415",
    'title':"PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering",
    'pretrain_dataset':{"ROCO":"/home/coder/projects/PubMedCLIP-main/PubMedCLIP/roco/ROCO/roco/roco-dataset-master/data",
                        "Mimic-CXR":"/home/coder/projects/PubMedCLIP-main/PubMedCLIP/roco/ROCO/roco/roco-dataset-master/data",
                        "MedICaT":"/home/coder/projects/PubMedCLIP-main/PubMedCLIP/roco/ROCO/roco/roco-dataset-master/data",
                        "PMC-VQA":"/home/coder/projects/PubMedCLIP-main/PubMedCLIP/roco/ROCO/roco/roco-dataset-master/data"},
    'pretrain_path':{"ROCO":"/home/coder/projects/MMBERT/pretrain.sh","Mimic-CXR":"/home/coder/projects/MMBERT/pretrain.sh",
                    "MedICaT":"/home/coder/projects/MMBERT/pretrain.sh","PMC-VQA":"/home/coder/projects/MMBERT/pretrain.sh"},
    "path":{
        "VQA-RAD":"/home/coder/projects/PMC-VQA/src/MedVInT_TD/train_downstream.sh",
        "PATHVQA":"/home/coder/projects/PMC-VQA/src/MedVInT_TD/train_downstream.sh",
         "SLAKE":"/home/coder/projects/PMC-VQA/src/MedVInT_TD/train_downstream.sh",
        "MED-VQA":"/home/coder/projects/PMC-VQA/src/MedVInT_TD/train_downstream.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD", "PATHVQA":"data_PathVQA","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE","MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019"}
}

MiniGPT4={
    'url':"https://arxiv.org/abs/2304.10592",
    'title':"MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models",
    "path":{
        "VQA-RAD":"/home/coder/projects/Med-VQA/run_vqarad.sh",
        "PATHVQA":"/home/coder/projects/Med-VQA/run_path.sh",
         "SLAKE":"/home/coder/projects/Med-VQA/run_slake.sh",
        "MED-VQA":"/home/coder/projects/Med-VQA/run_medvqa.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD", "PATHVQA":"data_PathVQA","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE","MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019"}
}

TCL={
    'url':"https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Vision-Language_Pre-Training_With_Triple_Contrastive_Learning_CVPR_2022_paper",
    'title':"Vision-language pre-training with triple contrastive learning",
    "path":{
        "VQA-RAD":"/home/coder/projects/TCL/run_rad.sh",
        "PATHVQA":"/home/coder/projects/TCL/run_path.sh",
         "SLAKE":"/home/coder/projects/TCL/run_slake.sh",
        "MED-VQA":"/home/coder/projects/TCL/run_med.sh"
        },
    "dataset":{"VQA-RAD":"data_RAD", "PATHVQA":"data_PathVQA","SLAKE":"/home/coder/projects/Med-VQA/data_SLAKE","MED-VQA":"/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019"}
}


models={
    'MEVF': MEVF,
    'CR':CR, 
    'MMBERT':MMBERT,
    'VQAMIX':VQAMix,
    'MMQ':MMQ,
    'CSMA':CSMA,
    'PTUNIFIER':PTUnifier,
    'TCL':TCL,
    'METER':METER,
    'MEDVINT':MedVInT,
    'MINIGPT-4':MiniGPT4,    
}
