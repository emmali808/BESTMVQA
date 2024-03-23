source activate mmbert
conda info -e
cd /home/coder/projects/MMBERT/vqarad

python train_vqarad.py --run_name hxj_test --data_dir /home/coder/projects/Med-VQA/data_SLAKE --save_dir /home/coder/projects/MMBERT/vqa_slake --use_pretrained --epochs $3 --lr $4 --batch_size $5 --record_id $7
