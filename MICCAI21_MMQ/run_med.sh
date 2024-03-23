source activate mmq
#conda info -e
cd /home/coder/projects/MICCAI21_MMQ
modelpath="saved_model_med"
ch="/"
modelpathout=$modelpath$ch$1
python3 main.py --use_VQA --VQA_dir $2 --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml.weights --maml_nums 0 --model $1 --lr $4 --seed 1342 --output $modelpathout --rnn $6 --record_id $7 --batch_size $5 --epochs $3
ch2="results-med"
outpath=$ch2$ch$1
python3 test.py --use_VQA --VQA_dir $2 --maml --autoencoder --feat_dim 64 --img_size 84 --maml_model_path pretrained_maml.weights --input $modelpathout --epoch _last --maml_nums 0 --model $1 --record_id $7 --output=$outpath 