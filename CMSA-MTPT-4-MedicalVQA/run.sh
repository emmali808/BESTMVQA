source activate cmsa
#conda info -e
cd /home/coder/projects/CMSA-MTPT-4-MedicalVQA
path="saved_model_"
modelpath=$path$2
ch="/"
modelpathout=$modelpath$ch$1
CUDA_VISIBLE_DEVICES=0 python3 main.py --model $1 --RAD_dir $2 --use_spatial --distmodal  --output $modelpathout --epochs $3 --lr $4 --batch_size $5 --rnn $6 --record_id $7
let modelepoch=$3-1
ch2="results"
outpath=$ch2$ch$2$1
CUDA_VISIBLE_DEVICES=0 python3 test.py --model $1 --RAD_dir $2 --distmodal --use_spatial --input $modelpathout --epoch $modelepoch --output $outpath --record_id $7