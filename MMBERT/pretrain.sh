source activate mmbert
cd /home/coder/projects/MMBERT/pretrain
python roco_train.py --run_name roco_pretrain --mlm_prob 0.15 --data_dir $1