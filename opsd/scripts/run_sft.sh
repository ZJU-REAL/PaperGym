export CUDA_VISIBLE_DEVICES=0

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps 8 \
    --main_process_port 19346 \
    sft_train.py \
    --model_name_or_path <YOUR_MODEL_PATH> \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --output_dir <YOUR_OUTPUT_DIR> \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 20000 \
    --logging_steps 5 \
    --save_steps 20 \
    --log_with none \
    --dataset_path "<YOUR_DATASET_PATH>"
