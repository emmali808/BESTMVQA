#!/usr/bin/env bash
python3 main.py --use_VQA --VQA_dir data_RAD --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml_pytorch_other_optimization_3shot_newmethod.pth --maml_nums 2,4 --model BAN --lr 0.001 --seed 1342 --output saved_models/MMQ_BAN_MEVF_vqaRAD
