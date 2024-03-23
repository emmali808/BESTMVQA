source activate MEVF
conda info -e
cd /home/coder/projects/MEVF/MICCAI19-MedVQA

python main.py --model $1 --use_RAD --RAD_dir $2 --maml --autoencoder --output saved_models/SAN_MEVF_SLAKE --epochs $3 --lr $4 --batch_size $5 --rnn $6 --record_id $7
let batchsize=$3-1
python test.py --model $1 --use_RAD --RAD_dir $2 --maml --autoencoder --input saved_models/SAN_MEVF_SLAKE --epoch ${batchsize} --output saved_models/SAN_MEVF_SLAKE --record_id $7
