# LLaVA-Med: Large Language and Vision Assistant for BioMedicine
**This model is developed based on the [model](https://github.com/microsoft/LLaVA-Med) proposed in this [paper](https://arxiv.org/abs/2306.00890).**

For our SOTA model on the five datasets. Please download the dataset at the link indicated by the file in the ASMVQA/README.MD and move to SystemDataset/dataset_name. 

### Prerequisites
Please install dependence package befor running the following command:
```
cd ./llava-med
conda activate llava-med
```

### Fine-tune on benchmark Med-VQA datasets
```
torchrun --nnodes=1 --nproc_per_node=8           --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /path/to/checkpoint_llava_med_instruct_60k_inline_mention \
    --data_path /path/to/eval/vqa_rad/train.json \
    --image_folder /path/to/eval/vqa_rad/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

### generate answers
```
python llava/eval/model_vqa_med.py --model-name /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad \
    --question-file path/to/eval/vqa_rad/test.json \
    --image-folder path/to/eval/vqa_rad/images \
    --answers-file /path/to/checkpoint_llava_med_instruct_60k_inline_mention/eval/fine_tuned/vqa_rad/test-answer-file.jsonl
```

### evaluate the generated answers
```
python llava/eval/run_eval_batch.py \
    --pred_file_parent_path /path/to/llava-med \
    --target_test_type test-answer-file
```