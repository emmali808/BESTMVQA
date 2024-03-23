source activate CR
conda info -e
cd /home/coder/projects/Med-VQA

python main.py --epochs $3 --lr $4 --batch_size $5 -rnn $6 --record_id $7 --gpu 0 --seed 88 --data_dir $2