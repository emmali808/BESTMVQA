source activate mmbert
conda info -e
cd /home/coder/projects/MMBERT/vqarad

python train_vqarad.py --run_name hxj_test --mixed_precision --use_pretrained --epochs $3 --lr $4 --batch_size $5 --record_id $7 --data_dir $2
