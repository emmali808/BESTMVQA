source activate MEVF
conda info -e
cd /home/coder/projects/MEVF/MICCAI19-MedVQA

python main02.py --model $1 --use_RAD --medvqa2019_dir $2 --maml --autoencoder --output saved_models/BAN_MEVF_medvqa2019 --epochs $3 --lr $4 --batch_size $5 --rnn $6 --record_id $7
let batchsize=$3-1
python test_medvqa.py --model $1 --use_RAD --medvqa2019_dir $2 --maml --autoencoder --input saved_models/BAN_MEVF_medvqa2019 --epoch ${batchsize} --record_id $7 --output results/BAN_MEVF_medvqa2019
