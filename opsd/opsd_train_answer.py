import os
import importlib

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer_answer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    log_with: str = field(
        default="wandb",
        metadata={
            "help": "Choose logging system: 'wandb' for Weights & Biases, 'swanlab' for SwanLab, or 'none' for no logging."
        },
    )
    dataset_path: str = field(
        default="",
        metadata={
            "help": "Path to local parquet file(s) for training data. Supports multiple files separated by commas."
        },
    )


def init_logging_system(project_name, run_name, config, log_system="wandb", output_dir=None):
    """
    Initialize logging system, supporting wandb and swanlab
    """
    import os
    if log_system == "wandb":
        if importlib.util.find_spec("wandb") is not None:
            import wandb
            # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
            if os.environ.get("LOCAL_RANK", "0") == "0":
                wandb.init(
                    entity=os.environ.get("WANDB_ENTITY"),  # Can be set via environment variable
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
            # Disable wandb if swanlab is used to prevent conflicts
            os.environ["WANDB_DISABLED"] = "true"
            # Only initialize swanlab on main process (LOCAL_RANK 0 or not set)
            if os.environ.get("LOCAL_RANK", "0") == "0":
                # Ensure output directory exists
                from pathlib import Path
                # Use subdirectory under current working directory
                swanlab_output_dir = Path.cwd() / "swanlab_runs"
                os.makedirs(swanlab_output_dir, exist_ok=True)
                
                swanlab.init(
                    project=project_name,
                    name=run_name,
                    description="OPSD Training Experiment",
                    config=config,
                    logdir=str(swanlab_output_dir)
                )
            return swanlab
        else:
            print("Warning: swanlab not installed, attempting to install...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "swanlab"])
                import swanlab
                # Disable wandb if swanlab is used to prevent conflicts
                os.environ["WANDB_DISABLED"] = "true"
                # Only initialize swanlab on main process (LOCAL_RANK 0 or not set)
                if os.environ.get("LOCAL_RANK", "0") == "0":
                    # Ensure output directory exists
                    from pathlib import Path
                    # Use subdirectory under current working directory
                    swanlab_output_dir = Path.cwd() / "swanlab_runs"
                    os.makedirs(swanlab_output_dir, exist_ok=True)
                    
                    swanlab.init(
                        project=project_name,
                        name=run_name,
                        description="OPSD Training Experiment",
                        config=config,
                        logdir=str(swanlab_output_dir)
                    )
                return swanlab
            except Exception as e:
                print(f"Failed to install swanlab: {e}")
                print("Continuing without swanlab logging")
                return None
    else:
        print("Logging disabled")
        # Explicitly disable wandb if logging is disabled
        os.environ["WANDB_DISABLED"] = "true"
        return None


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Run Configuration
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/<user>/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to run config name if enabled
        if script_args.fixed_teacher:
            full_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"Run Name: {full_run_config}")
    print(f"Log System: {script_args.log_with}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # Logging Configuration
    ################
    # Set report_to early to prevent conflicts with wandb/swanlab initialization
    if script_args.log_with == "swanlab":
        training_args.report_to = ["swanlab"]
    elif script_args.log_with == "none":
        training_args.report_to = []
    elif script_args.log_with == "wandb":
        # Ensure WANDB_DISABLED is not set when using wandb
        if "WANDB_DISABLED" in os.environ:
            del os.environ["WANDB_DISABLED"]

    ################
    # Logging System Initialization
    ################
    # Initialize logging system based on the log_with argument
    logger = init_logging_system(
        project_name=training_args.wandb_project or "OPSD",
        run_name=full_run_config,
        config={
            "model_name": model_args.model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "num_train_epochs": training_args.num_train_epochs,
            "max_completion_length": training_args.max_completion_length,
            "temperature": training_args.temperature,
            "beta": training_args.beta,
            "lmbda": training_args.lmbda,
            "max_length": training_args.max_length,
            "use_peft": model_args.use_peft,
            "lora_r": model_args.lora_r if model_args.use_peft else None,
            "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "num_processes": num_processes,
            "use_tinker_loss": script_args.use_tinker_loss,
            "fixed_teacher": script_args.fixed_teacher,
            "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
            "use_ema_teacher": script_args.use_ema_teacher,
            "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
        },
        log_system=script_args.log_with,
        output_dir=training_args.output_dir,
    )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    # Load dataset from local parquet files
    import pyarrow.parquet as pq
    import pandas as pd
    from datasets import Dataset

    # Parse dataset_path: support multiple files separated by commas
    if script_args.dataset_path:
        parquet_files = [f.strip() for f in script_args.dataset_path.split(",") if f.strip()]
    else:
        # Fallback to default if not provided (should not happen when called from scripts)
        parquet_files = []

    if not parquet_files:
        raise ValueError("--dataset_path is required. Please provide path(s) to parquet file(s). Multiple files can be separated by commas.")

    # Load and concatenate the parquet files
    dataframes = []
    for file in parquet_files:
        table = pq.read_table(file)
        df = table.to_pandas()
        dataframes.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(combined_df)
    
    # Split dataset if needed (using a simple split for now)
    train_test_split = dataset.train_test_split(test_size=0.05)  # Small test set
    train_dataset = train_test_split['train']

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        log_with=script_args.log_with,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)