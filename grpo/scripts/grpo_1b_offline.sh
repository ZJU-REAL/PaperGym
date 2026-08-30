#!/bin/bash

set -x

# Get parameters
nproc_per_node=2
save_path=<YOUR_SAVE_PATH>

# Specify GPU to use
export CUDA_VISIBLE_DEVICES=0,1

# Set wandb and swanlab offline mode
export WANDB_MODE=offline

echo "[CUDA_VISIBLE_DEVICES]: $CUDA_VISIBLE_DEVICES"

# Build training command
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=<YOUR_TRAIN_DATASET_PATH> \
    data.val_files=<YOUR_VAL_DATASET_PATH> \
    data.train_batch_size=32 \
    data.max_prompt_length=2500 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    custom_reward_function.path=<YOUR_REWARD_FUNCTION_PATH> \
    custom_reward_function.name="paper_rl_reward" \
    actor_rollout_ref.model.path=<YOUR_MODEL_PATH> \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$nproc_per_node \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=4600 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_local_dir=$save_path \
    trainer.project_name=paper-rl \
    trainer.experiment_name=<YOUR_EXPERIMENT_PATH> \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.logger=['console','swanlab'] \
    $@