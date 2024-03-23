source activate /home/coder/miniconda/envs/meter02
cd /home/coder/projects/METER/
python run.py with data_root=/home/coder/projects/METER/data/vqa_med2019 num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=$3 load_path=meter_clip16_288_roberta_pretrain.ckpt clip16 text_roberta image_size=144 clip_randaug max_epoch=$1 learning_rate=$2 record_id=$4 vocab_size=10000 task_vqa_medvqa_2019

python run.py with data_root=/home/coder/projects/METER/data/vqa_med2019 num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=16 load_path=/home/coder/projects/METER/result/task_vqa_medvqa_2019_seed0_from_meter_clip16_288_roberta_pretrain/version_12/checkpoints/last.ckpt clip16 text_roberta image_size=144 clip_randaug vocab_size=10000  test_only=True clip16 task_vqa_medvqa_2019 max_epoch=$1 learning_rate=$2 record_id=$4
