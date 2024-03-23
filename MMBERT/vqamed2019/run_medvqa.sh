source activate mmbert
conda info -e
cd /home/coder/projects/MMBERT/vqamed2019

python train.py --run_name  hxj_test --num_vis 5 --epochs $3 --lr $4 --batch_size $5 --record_id $7 --data_dir $2
