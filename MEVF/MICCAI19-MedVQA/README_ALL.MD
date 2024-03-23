## vqa-ovqa
# SAN
# 微调
<!-- python main.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --output saved_models/SAN_MEVF_OVQA --epochs 20 --lr 0.001 --batch_size 32 -->
python main.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --output saved_models/SAN_MEVF_OVQA --epochs 20 --lr 0.001 --batch_size 64
# 测试
python test.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --input saved_models/SAN_MEVF_OVQA --epoch 19 --output results/SAN_MEVF

# BAN
# 微调
python main.py --model BAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --output saved_models/BAN_MEVF_OVQA --epochs 25 --lr 0.001 --batch_size 128
# 测试
python test.py --model BAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --maml --autoencoder --input saved_models/BAN_MEVF_OVQA --epoch 24 --output results/BAN_MEVF

## vqa-rad
# 微调
python main.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_RAD --maml --autoencoder --output saved_models/SAN_MEVF --epochs 20 --lr 0.001 --batch_size 32

# 测试

python test.py --model SAN --use_RAD --RAD_dir /home/coder/projects/SystemDataset/data_RAD --maml --autoencoder --input saved_models/SAN_MEVF --epoch 19 --output results/SAN_MEVF


## medvqa2019数据集
# 微调
# CUDA_VISIBLE_DEVICES=1

python main02.py --model BAN --use_RAD --medvqa2019_dir data_Med/VQA-Med-2019/ --maml --autoencoder --output saved_models/BAN_MEVF_medvqa2019 --epochs 1

# 测试

python test_medvqa.py --model BAN --use_RAD --medvqa2019_dir data_Med/VQA-Med-2019/ --maml --autoencoder --input saved_models/BAN_MEVF_medvqa2019 --epoch 0 --output results/BAN_MEVF_medvqa2019


## slake数据集

# SAN!!!
# 微调
# epochs 必须设置为大于15 才会保存模型,也可以通过调整代码修改此设置
python main.py --model SAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --output saved_models/SAN_MEVF_SLAKE --epochs 40

# 测试
# epoch指定使用哪一个epoch的模型
python test.py --model SAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --input saved_models/SAN_MEVF_SLAKE --epoch 15 --output saved_models/SAN_MEVF_SLAKE --record_id 1

# BAN!!!
# 微调
# epochs 必须设置为大于15 才会保存模型,也可以通过调整代码修改此设置
python main.py --model BAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --output saved_models/BAN_MEVF_SLAKE --epochs 1

# 测试
# epoch指定使用哪一个epoch的模型
python test.py --model BAN --use_RAD --RAD_dir /home/coder/projects/Med-VQA/data_SLAKE --maml --autoencoder --input saved_models/BAN_MEVF_SLAKE --epoch 0 --output saved_models/BAN_MEVF_SLAKE



## PATH数据集

# 微调
# 这个基线的数据集存放在/dev/shm/data_PATH , 因为数据预处理方式不太一样
# SAN!!!
python main.py --model SAN --use_RAD --RAD_dir /dev/shm/data_PATH --maml --autoencoder --output saved_models/SAN_MEVF_PATH --epochs 30

# 测试
# epoch指定使用哪一个epoch的模型
python test.py --model SAN --use_RAD --RAD_dir /dev/shm/data_PATH --maml --autoencoder --input saved_models/SAN_MEVF_PATH --epoch 19 --output saved_models/SAN_MEVF_PATH --record_id 1


# BAN!!!
# 微调
# epochs 必须设置为大于15 才会保存模型,也可以通过调整代码修改此设置
python main.py --model BAN --use_RAD --RAD_dir /dev/shm/data_PATH --maml --autoencoder --output saved_models/BAN_MEVF_PATH --epochs 20

# 测试
# epoch指定使用哪一个epoch的模型
python test.py --model BAN --use_RAD --RAD_dir /dev/shm/data_PATH --maml --autoencoder --input saved_models/BAN_MEVF_PATH --epoch 19 --output saved_models/BAN_MEVF_PATH
