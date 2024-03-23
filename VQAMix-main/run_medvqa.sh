source activate VQAMix
#conda info -e
cd /home/coder/projects/VQAMix-main
path="saved_model_Med2019"
modelpath=$path
ch="/"
modelpathout=$modelpath$ch$1
python main.py --model $1 --use_RAD --RAD_dir $2 --output $modelpathout --lr $4 --epochs $3 --batch_size $5 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3 --rnn $6 --record_id $7
ch2="results"
dataname="Med2019"
outpath=$ch2$ch$dataname
let modelepoch=$3-1
python test.py --model $1 --use_RAD --RAD_dir $2 --input $modelpathout --output=$outpath --record_id $7 --epoch $modelepoch


