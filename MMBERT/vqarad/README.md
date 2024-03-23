## 自己调整epochs大小
# ovqa数据集
# 微调
# epoch设置为1，注意修改
python train_vqarad.py --run_name ovqa_test --epochs 80 --data_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --save_dir /home/coder/projects/MMBERT/vqa_ovqa --use_pretrained --batch_size 64
# 测试

python test_vqarad.py --run_name ovqa_test  --data_dir /home/coder/projects/SystemDataset/data_OVQA_as_RAD --fine_tune_model_dir /home/coder/projects/MMBERT/vqa_ovqa/ovqa_test_acc.pt

# vaqrad数据集
```
python train_vqarad.py --run_name hxj_test --epochs 1
```

# slake数据集
# batch_size改w为8,默认值16跑不动
# 微调
python train_vqarad.py --run_name hxj_test --epochs 100 --data_dir /home/coder/projects/Med-VQA/data_SLAKE --save_dir /home/coder/projects/MMBERT/vqa_slake --use_pretrained --batch_size 8
# 测试

python test_vqarad.py --run_name hxj_test  --data_dir /home/coder/projects/Med-VQA/data_SLAKE --fine_tune_model_dir /home/coder/projects/MMBERT/vqa_slake/hxj_test_test_acc.pt


# path数据集
# 微调
# batch_size改w为8,默认值16跑不动
python train_vqarad.py --run_name hxj_test --epochs 1 --data_dir /home/coder/projects/Med-VQA/data_PATH --save_dir /home/coder/projects/MMBERT/vqa_path --use_pretrained --batch_size 8

# 测试

python test_vqarad.py --run_name hxj_test  --data_dir /home/coder/projects/Med-VQA/data_PATH --fine_tune_model_dir /home/coder/projects/MMBERT/vqa_path/hxj_test_test_acc.pt