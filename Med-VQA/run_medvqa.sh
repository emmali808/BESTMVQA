source activate CR
conda info -e
cd /home/coder/projects/Med-VQA

python main03.py --autoencoder --maml --dataset $2 --data_dir /home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019 --epochs $3 --lr $4 --batch_size $5 -rnn $6 --record_id $7

