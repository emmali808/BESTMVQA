source activate /home/coder/miniconda/envs/meter02
python /home/coder/projects/METER/arrow_for_robot.py
python /home/coder/projects/METER/run.py with data_root=/home/coder/projects/METER/data/vqa_robot_demo num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=2 load_path=/home/coder/projects/METER/result/finetune_vqa_seed0_from_meter_clip16_288_roberta_pretrain/version_45/checkpoints/last.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=1 vocab_size=10000 test_only=True
python /home/coder/projects/METER/save_ans.py