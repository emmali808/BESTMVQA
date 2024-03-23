source activate VQAMix
#conda info -e
cd /home/coder/projects/VQAMix-main
path="saved_model_"
modelpath=$path$2
ch="/"
modelpathout=$modelpath$ch$1
CUDA_VISIBLE_DEVICES=0 python3 main.py --model $1 --use_RAD --RAD_dir $2 --output $modelpathout --batch_size $5 --epochs $3 --lr $4 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 0 --rnn $6 --record_id $7
ch2="results"
outpath=$ch2$ch$2$1
let modelepoch=$3-1
CUDA_VISIBLE_DEVICES=0 python3 test.py --model $1 --use_RAD --RAD_dir $2 --input $modelpathout --output=$outpath --record_id $7 --epoch $modelepoch
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model BAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/BAN_MEVF-PATH-VQAMix-0 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 0
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/BAN_MEVF-PATH-VQAMix-0 --output="results-path"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model BAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/BAN_MEVF-PATH-VQAMix-1 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 1
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/BAN_MEVF-PATH-VQAMix-1 --output="results-path"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model BAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/BAN_MEVF-PATH-VQAMix-2 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 2
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/BAN_MEVF-PATH-VQAMix-2 --output="results-path"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model BAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/BAN_MEVF-PATH-VQAMix-3 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/BAN_MEVF-PATH-VQAMix-3 --output="results-path"
# CUDA_VISIBLE_DEVICES=0 python3 main.py --model BAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/BAN_MEVF-PATH-VQAMix-4 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 4
# CUDA_VISIBLE_DEVICES=0 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/BAN_MEVF-PATH-VQAMix-4 --output="results-path"
# CUDA_VISIBLE_DEVICES=0 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/SAN_MEVF-PATH-VQAMix-0 --output="results-path"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/SAN_MEVF-PATH-VQAMix-1 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 1
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/SAN_MEVF-PATH-VQAMix-1 --output="results-path"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/SAN_MEVF-PATH-VQAMix-2 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 2
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/SAN_MEVF-PATH-VQAMix-2 --output="results-path"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/SAN_MEVF-PATH-VQAMix-3 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/SAN_MEVF-PATH-VQAMix-3 --output="results-path"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_PathVQA --output saved_model_path/SAN_MEVF-PATH-VQAMix-4 --batch_size 128 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 4
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_model_path/SAN_MEVF-PATH-VQAMix-4 --output="results-path"
