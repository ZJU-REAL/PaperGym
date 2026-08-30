export CUDA_VISIBLE_DEVICES=0

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps 8 \
    --main_process_port 12953 \
    opsd_train_answer.py \
    --model_name_or_path "<YOUR_MODEL_PATH>" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --output_dir "<YOUR_OUTPUT_DIR>" \
    --run_config qwen3_1b_opsd_answer \
    --num_train_epochs 5 \
    --max_completion_length 2048 \
    --save_steps 50 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --log_with swanlab \
    --dataset_path "<YOUR_DATASET_PATH>"
