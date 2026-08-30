import os
import importlib

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    SFTTrainer,
    SFTConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with local dataset and logging options."""

    dataset_path: str = field(
        default="",
        metadata={
            "help": "Path to local parquet file for training data."
        },
    )
    log_with: str = field(
        default="wandb",
        metadata={
            "help": "Choose logging system: 'wandb' for Weights & Biases, 'swanlab' for SwanLab, or 'none' for no logging."
        },
    )


def make_format_fn(tokenizer):
    """
    Returns a formatting function that applies the chat template,
    matching the eval prompt format exactly.
    """

    def format_example(example):
        messages = [
            {
                "role": "user",
                "content": f"{example['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            },
            {
                "role": "assistant",
                "content": example["solution"]["reference_answer"],
            },
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    return format_example


def init_logging_system(project_name, run_name, config, log_system="wandb"):
    """
    Initialize logging system, supporting wandb and swanlab
    """
    if log_system == "wandb":
        if importlib.util.find_spec("wandb") is not None:
            import wandb
            if os.environ.get("LOCAL_RANK", "0") == "0":
                wandb.init(
                    entity=os.environ.get("WANDB_ENTITY"),
                    project=project_name,
                    name=run_name,
                    config=config,
                )
            return wandb
        else:
            print("Warning: wandb not installed, skipping wandb initialization")
            return None
    elif log_system == "swanlab":
        if importlib.util.find_spec("swanlab") is not None:
            import swanlab
            os.environ["WANDB_DISABLED"] = "true"
            if os.environ.get("LOCAL_RANK", "0") == "0":
                from pathlib import Path
                swanlab_output_dir = Path.cwd() / "swanlab_runs"
                os.makedirs(swanlab_output_dir, exist_ok=True)
                swanlab.init(
                    project=project_name,
                    name=run_name,
                    description="SFT Training Experiment",
                    config=config,
                    logdir=str(swanlab_output_dir),
                )
            return swanlab
        else:
            print("Warning: swanlab not installed, attempting to install...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "swanlab"])
                import swanlab
                os.environ["WANDB_DISABLED"] = "true"
                if os.environ.get("LOCAL_RANK", "0") == "0":
                    from pathlib import Path
                    swanlab_output_dir = Path.cwd() / "swanlab_runs"
                    os.makedirs(swanlab_output_dir, exist_ok=True)
                    swanlab.init(
                        project=project_name,
                        name=run_name,
                        description="SFT Training Experiment",
                        config=config,
                        logdir=str(swanlab_output_dir),
                    )
                return swanlab
            except Exception as e:
                print(f"Failed to install swanlab: {e}")
                print("Continuing without swanlab logging")
                return None
    else:
        print("Logging disabled")
        os.environ["WANDB_DISABLED"] = "true"
        return None


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Run Configuration
    ################
    model_name = model_args.model_name_or_path.split("/")[-1]

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    full_run_name = (
        f"SFT_{model_name}_"
        f"lr{lr_str}_"
        f"bs{effective_batch_size}_"
        f"ep{training_args.num_train_epochs}"
    )

    ################
    # Logging Configuration
    ################
    if script_args.log_with == "swanlab":
        training_args.report_to = ["swanlab"]
    elif script_args.log_with == "none":
        training_args.report_to = []
    elif script_args.log_with == "wandb":
        if "WANDB_DISABLED" in os.environ:
            del os.environ["WANDB_DISABLED"]

    ################
    # Logging System Initialization
    ################
    logger = init_logging_system(
        project_name="SFT-Math",
        run_name=full_run_name,
        config={
            "model_name": model_args.model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "num_train_epochs": training_args.num_train_epochs,
            "max_seq_length": training_args.max_length,
            "use_peft": model_args.use_peft,
            "lora_r": model_args.lora_r if model_args.use_peft else None,
            "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "num_processes": num_processes,
        },
        log_system=script_args.log_with,
    )

    ################
    # Model & Tokenizer
    ################
    import torch

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",  # Use right padding for SFT
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    if not script_args.dataset_path:
        raise ValueError("--dataset_path is required. Please provide a path to a local parquet file.")

    # Load from local parquet file
    import pyarrow.parquet as pq
    import pandas as pd
    from datasets import Dataset

    table = pq.read_table(script_args.dataset_path)
    df = table.to_pandas()
    dataset = Dataset.from_pandas(df)
    print(f"Loaded dataset from {script_args.dataset_path}: {len(dataset)} examples")

    dataset = dataset.map(make_format_fn(tokenizer))

    # Take 1% of train for evaluation if no eval split exists
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
