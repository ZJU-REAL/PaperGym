# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from trl.data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from trl.experimental.gold.gold_config import GOLDConfig
from data_collator_combine import SelfDistillationDataCollator


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


class EMAUpdateCallback(TrainerCallback):
    """Callback class for updating EMA teacher weights, executed after each optimizer step."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only update when optimizer actually steps (at end of gradient accumulation period)
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class GOLDVLLMSyncCallback(TrainerCallback):
    """Callback class for syncing model weights to vLLM, sync when safe."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step, when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a gradient sync step
            # This happens at the end of gradient accumulation period
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class OPSDTrainer(SFTTrainer):
    """
    OPSD (OPen Source Distillation) Trainer class.
    Based on SFTTrainer extension, implementing self-distillation training logic
    """
    _tag_names = ["trl", "opsd"]
    _name = "OPSD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,      # Use Thinking Machines style reverse KL loss
        fixed_teacher: bool = False,                  # Use initial policy as fixed teacher
        reason_first: bool = False,                   # Teacher first reasons rationally
        top_k_loss: int | None = None,                # Limit JSD loss to top-k tokens of teacher distribution
        jsd_token_clip: float | None = None,          # Maximum clipping value for JSD loss per token
        use_ema_teacher: bool = False,                # Use exponential moving average of student weights as teacher
        ema_decay: float = 0.999,                     # EMA decay factor
        log_with: str = "wandb",                      # Specify logging system
    ):
        # Save model path information
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Custom data collator for self-distillation
        if data_collator is None:
            data_collator = SelfDistillationDataCollator(
                tokenizer=processing_class, max_length=args.max_length, reason_first=reason_first
            )

        # Call parent class initialization method
        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        # If dropout is disabled
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        # Save training parameters
        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.use_thinking_machines_loss = use_thinking_machines_loss
        self.fixed_teacher = fixed_teacher
        self.reason_first = reason_first
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self.log_with = log_with  # Save logging system settings
        self._ema_params = None  # Lazy initialization at first optimizer step

        # Validate fixed_teacher option
        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forward passes."
            )

        # Ensure EMA teacher and fixed teacher cannot be enabled simultaneously
        if self.use_ema_teacher and self.fixed_teacher:
            raise ValueError(
                "use_ema_teacher=True and fixed_teacher=True are mutually exclusive teacher strategies."
            )

        # Print mode information based on settings
        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'='*80}")
            print("EMA TEACHER MODE ENABLED")
            print(f"EMA decay: {self.ema_decay}")
            print("Teacher is an exponential moving average of the student weights.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'='*80}\n")

        if self.fixed_teacher:
            print(f"\n{'='*80}")
            print("FIXED TEACHER MODE ENABLED")
            print("Teacher will use the initial policy (base model without LoRA adapters)")
            print("Student will update with LoRA adapters")
            print(f"{'='*80}\n")

        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASON FIRST MODE ENABLED")
            print("Teacher will first reason about the privileged solution, then evaluate student's response")
            print(f"{'='*80}\n")

        # Track per-step loss statistics (for logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        # Track generation outputs to save
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5  # Save every 5 steps

        # Initialize generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Generation config for reasoning phase
        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._metrics["train"]["seq_kl_divergences"] = []  # Add list for sequence KL divergence
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the number of forward passes optimized per step. This ensures we only record the final optimization step
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        # Initialize vLLM related configuration
        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Ensure tensor_parallel_size divides world_size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM needs environment variables set in distributed training
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Provide same seed for TP group to ensure same sampling results
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process loads model weights. This may cause desynchronization and DeepSpeed to hang. To prevent this,
                # we synchronize all processes after vLLM fully initializes
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

    def _set_signature_columns_if_needed(self):
        """Set signature columns (if needed)"""
        super()._set_signature_columns_if_needed()
        required_columns = [
            "problem",      # Question column
            "solution",     # Solution column
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
    ):
        """
        Compute generalized Jensen-Shannon divergence loss for knowledge distillation using F.kl_div.
        
        Args:
        - student_logits: Student model log-probabilities, shape (batch_size, sequence_length, vocab_size)
        - teacher_logits: Teacher model log-probabilities, shape (batch_size, sequence_length, vocab_size)
        - labels: Label tensor, -100 indicates padding tokens to ignore in loss computation
        - beta: Interpolation coefficient, between 0 and 1, default 0.5
        - temperature: Softmax temperature, default 1.0
        - reduction: Loss reduction method, default 'batchmean'
        - top_k: If set, restrict loss to top-k tokens in teacher distribution
        - token_clip: If set, clip divergence per token to prevent style tokens from dominating gradient
        """
        
        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling before computing probabilities from logits
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

            if top_k is not None and top_k > 0:
                # Limit to top-k tokens of teacher logits and renormalize
                # Shape: [batch, seq_len, top_k]
                _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
                teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

            # Compute log probabilities for student, probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute log of mixture distribution
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergence using F.kl_div
            # PyTorch differs from standard mathematical definition, so probability distribution order is opposite to paper
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute generalized Jensen-Shannon divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Per-token clipping: limit divergence value for each token
        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        # Apply mask
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def _update_ema(self):
        """
        Update EMA parameters after optimizer step.

        On first call, lazily initializes EMA state as an exact copy of current (trainable) model parameters,
        then returns without applying decay step. Subsequent calls apply: ema = decay * ema + (1 - decay) * student.

        Only tracks trainable parameters (i.e., LoRA adapter weights for PEFT models or all parameters for full fine-tuning).

        ZeRO-3 Note: When using ZeRO-3, each rank only holds a shard of each parameter.
        We use `deepspeed.zero.GatheredParameters` (read-only, modifier_rank=None) so that
        each rank can see the full parameter tensor for snapshot/update EMA.
        Therefore the EMA tensor is a full-size copy, which is what `_ema_teacher_context` needs when replacing student weights with EMA values.
        """
        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Detect ZeRO-3 (same pattern used elsewhere)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            # Get all parameters requiring gradients
            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            # modifier_rank=None -> read-only gather; restore original partition on exit
            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    # Initialize EMA parameters
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                        f"(decay={decay})"
                    )
                    return  # First call: only initialize, do not update decay

                # Update EMA parameters
                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
        else:
            if self._ema_params is None:
                # Lazy initialization: snapshot current weights as initial EMA state
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                    f"(decay={decay})"
                )
                return  # First call: only initialize, do not update decay

            # Update EMA parameters
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                # Move EMA buffer to same device as live parameters (handle multi-GPU setup)
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        """
        Context manager to temporarily load EMA weights for teacher forward pass.

        Swaps `param.data` for each tracked (trainable) parameter with its EMA counterpart,
        executes the body (teacher forward), then unconditionally restores student weights.
        Safe to use inside `torch.no_grad()`. If EMA is not yet initialized (step 0),
        this is a no-op, using current student weights.

        ZeRO-3 Note: Direct `param.data` assignment bypasses ZeRO-3's sharding lifecycle and
        corrupts its internal state, causing size mismatch errors during gradient checkpoint recomputation.
        When ZeRO-3 is active, we therefore wrap the swap in
        `deepspeed.zero.GatheredParameters` so that before we touch them,
        parameters are fully instantiated on each rank, and ZeRO-3 cleanly repartitions on context exit.
        """
        if self._ema_params is None:
            yield  # EMA not initialized; fallback to current weights
            return

        unwrapped = self.accelerator.unwrap_model(model)

        # Detect ZeRO-3 (same pattern used elsewhere)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            params_list = list(name_to_param.values())

            # modifier_rank=0 causes ZeRO-3 to repartition from rank-0's param.data on exit,
            # which will be the restored student weights
            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        if name in saved:
                            param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def compute_kl_divergence(self, student_logits, teacher_logits, temperature=1.0, labels=None):
        """
        Compute KL divergence between student and teacher models.
        
        Returns:
            overall_avg_kl: Average KL divergence over all valid tokens
            seq_avg_kl: List of average KL divergence per sequence
        """
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        
        # Compute log softmax for numerically stable KL divergence calculation
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
        # Where P is teacher distribution, Q is student distribution
        # Result shape: [batch_size, seq_len, vocab_size]
        kl_div_per_vocab = teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
        
        # Sum over vocabulary dimension to get KL divergence per token
        # Shape: [batch_size, seq_len]
        kl_div = kl_div_per_vocab.sum(dim=-1)
        
        # If labels provided, only compute average KL divergence for non-padding tokens
        if labels is not None:
            mask = labels != -100  # shape: [batch_size, seq_len]
            
            # Only compute KL divergence for non-padding tokens
            kl_div_masked = kl_div[mask]  # shape: [num_valid_tokens]
            
            # Compute average KL divergence per sequence
            batch_size, seq_len = labels.shape
            seq_avg_kl = []
            
            for i in range(batch_size):
                valid_tokens = mask[i].sum().item()
                if valid_tokens > 0:
                    # Compute average KL divergence for current sequence
                    seq_kl = kl_div[i][mask[i]].mean().item()
                    seq_avg_kl.append(seq_kl)
                else:
                    seq_avg_kl.append(0.0)
            
            # Compute overall average KL divergence for all valid tokens
            total_valid_tokens = mask.sum().item()
            overall_avg_kl = kl_div_masked.sum().item() / total_valid_tokens if total_valid_tokens > 0 else 0.0
            
            return overall_avg_kl, seq_avg_kl
        else:
            # If no labels, compute average KL divergence for all tokens
            return kl_div.mean().item(), [kl_div.mean().item()]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute self-distillation loss, extracting log-probabilities in a memory-efficient way.

        Memory optimization: Immediately extract only the needed log-probabilities and release large tensors.
        """
        # Get batch-level prompt length
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # === Student forward - immediately extract log probabilities ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )

        # Immediately extract what we need and convert to log probabilities
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]

        if self.use_thinking_machines_loss:
            # For reverse KL, we only need log probabilities of sampled tokens
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            student_log_probs_sampled = torch.gather(
                student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            del student_logits, student_log_probs  # Release immediately!
        else:
            # For JSD, keep logits (temperature will be applied in generalized_jsd_loss)
            student_logits_for_loss = student_logits
            del student_logits

        # Release full output (but keep reference for return_outputs)
        if return_outputs:
            # Create a minimal output object to return (only loss, no logits)
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        # === Teacher forward - immediately extract log probabilities ===
        # Select teacher context based on mode:
        # use_ema_teacher  → Temporarily switch to EMA weights
        # fixed_teacher    → Disable LoRA adapters (base model = initial policy)
        # Default (dynamic) -> no-op, use current student weights
        if self.use_ema_teacher:
            adapter_context = self._ema_teacher_context(model)
        elif self.fixed_teacher and is_peft_model(model):
            adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            adapter_context = nullcontext()

        with torch.no_grad(), adapter_context:
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]

            if self.use_thinking_machines_loss:
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del teacher_logits, teacher_log_probs  # Release immediately!
            else:
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

            del outputs_teacher
            empty_cache()

        # === Compute KL divergence and record ===
        if not self.use_thinking_machines_loss:
            with torch.no_grad():
                # Prepare labels for KL divergence calculation (adjust length dimension)
                adjusted_labels = inputs["labels"][:, student_prompt_len:]
                
                kl_div_overall, seq_kl_divs = self.compute_kl_divergence(
                    student_logits_for_loss.detach(),
                    teacher_logits_for_loss.detach(),
                    temperature=self.temperature,
                    labels=adjusted_labels
                )
                
                # Record overall average KL divergence
                self._metrics["train"]["kl_divergence"].append(kl_div_overall)
                
                # Record average KL divergence per sequence
                self._metrics["train"]["seq_kl_divergences"].extend(seq_kl_divs)

        # === Compute loss using small tensors ===
        if self.use_thinking_machines_loss:
            # Thinking Machines uses RL-style policy gradient:
            # Advantage = log π_teacher(x) - log π_student(x)
            # Loss = -E[advantage * log π_student(x)]
            #
            # Key: advantage must be detached to prevent gradient flow through it.
            # We want: ∇θ L = -E[A(x) * ∇θ log π_student(x)]
            # Not: ∇θ L = -E[(T(x) - S(x)) * ∇θ S(x)] where both are differentiated

            advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

            # Apply mask before computing loss
            if shifted_labels is not None:
                mask = shifted_labels != -100
                advantage = advantage[mask]
                student_log_probs_sampled_masked = student_log_probs_sampled[mask]
            else:
                student_log_probs_sampled_masked = student_log_probs_sampled

            # Policy gradient loss: -advantage * log π_student
            # Negative because we minimize loss (gradient descent) but want to maximize reward
            loss = -(advantage * student_log_probs_sampled_masked).mean()

            del (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                advantage,
                student_log_probs_sampled_masked,
            )
        else:
            # Temperature applied inside generalized_jsd_loss
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,  # Let the function handle temperature
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )
            del student_logits_for_loss, teacher_logits_for_loss

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return (loss, minimal_output)
        else:
            return loss

    def generate_teacher_reasoning(
        self, model, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher reasoning for the solution."""
        if self.use_vllm:
            # Use vLLM for fast inference generation
            return self._generate_teacher_reasoning_vllm(teacher_reasoning_prompts)
        else:
            # Use transformers generation (slower)
            with torch.no_grad():
                # Temporarily enable KV cache
                original_use_cache = model.config.use_cache
                original_gen_use_cache = self.reasoning_generation_config.use_cache

                model.config.use_cache = True
                self.reasoning_generation_config.use_cache = True

                # If fixed_teacher=True, disable LoRA adapters
                adapter_context = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self.fixed_teacher and is_peft_model(model)
                    else nullcontext()
                )

                try:
                    with adapter_context:
                        reasoning_outputs = model.generate(
                            input_ids=teacher_reasoning_prompts,
                            attention_mask=teacher_reasoning_attention_mask,
                            generation_config=self.reasoning_generation_config,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                        reasoning_ids = reasoning_outputs.sequences
                finally:
                    model.config.use_cache = original_use_cache
                    self.reasoning_generation_config.use_cache = original_gen_use_cache

                return reasoning_ids

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs only from student prompts."""
        import time

        start_time = time.time()

        # Temporarily enable KV cache if disabled during training
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache

        model.config.use_cache = True
        generation_config.use_cache = True

        print(f"\n{'='*80}")
        print(f"GENERATION DEBUG INFO:")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Model config use_cache: {model.config.use_cache}")
        print(f"  Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
        print(f"  Generation config use_cache: {generation_config.use_cache}")
        print(f"  Batch size: {inputs['student_prompts'].shape[0]}")
        print(f"  Prompt length: {inputs['student_prompts'].shape[1]}")
        print(f"  Max new tokens: {generation_config.max_new_tokens}")
        print(f"{'='*80}\n")

        # Generate outputs, only for student prompts
        try:
            generated_outputs = model.generate(
                input_ids=inputs["student_prompts"],
                attention_mask=inputs.get("student_prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            # Get generated token IDs
            generated_tokens = generated_outputs.sequences
        finally:
            # Restore original settings
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        num_prompts = generated_tokens.shape[0]
        total_completion_tokens = generated_tokens.shape[1] - inputs["student_prompts"].shape[1]
        num_tokens = total_completion_tokens * num_prompts
        avg_completion_length = total_completion_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {num_tokens}, avg length: {avg_completion_length}, speed: {tokens_per_sec:.1f} tok/s"
        )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts using vLLM."""
        import time

        device = self.accelerator.device

        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )
        # Remove padding token text (if exists), because vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        # Also decode prompts with special tokens for logging
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )

        # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        # target_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        # prompts_text = [p.replace(target_system_prompt, system_prompt) for p in prompts_text]
        # Add system prompt for prompts

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p, presence_penalty not in generation_config, get from trainer parameters
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0

        # Start timing vLLM generation
        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather and flatten prompts from all ranks in TP group
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Get completions for this rank's slice in TP group
                # Each rank generates all outputs - we only keep our own share
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # Compute and print vLLM generation statistics
        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"vLLM generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {total_completion_tokens}, avg length: {avg_completion_length:.1f}, speed: {tokens_per_sec:.1f} tok/s"
        )

        # We need to merge prompts and completions to get new_input_ids
        # Retokenize prompts to get prompt IDs on correct device and format
        # Tokenize using prompts_text_for_vllm (no special tokens), because vLLM expects raw text
        # Ensure add_special_tokens=False, because vLLM typically handles raw text prompts
        # Compute max prompt length, ensure it is positive
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # If longer than max_completion_length, truncate
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # If shorter than max_completion_length, pad
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already correct length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Extract completion text from generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _generate_teacher_reasoning_vllm(
        self, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher reasoning using vLLM."""
        import time

        device = self.accelerator.device

        # Decode prompts for vLLM
        prompts_text = self.processing_class.batch_decode(
            teacher_reasoning_prompts,
            skip_special_tokens=True,
        )
        if self.processing_class.pad_token:
            prompts_text = [p.replace(self.processing_class.pad_token, "") for p in prompts_text]

        max_reasoning_length = self.reasoning_generation_config.max_new_tokens
        temperature = self.reasoning_generation_config.temperature
        top_k = (
            self.reasoning_generation_config.top_k
            if self.reasoning_generation_config.top_k and self.reasoning_generation_config.top_k > 0
            else -1
        )
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0

        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_reasoning_length,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]

        elif self.vllm_mode == "colocate":
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_reasoning_length,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)

        elapsed_time = time.time() - start_time
        total_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        print(
            f"vLLM teacher reasoning generation done - elapsed: {elapsed_time:.2f}s, prompts: {num_prompts}, tokens: {total_tokens}, speed: {total_tokens/elapsed_time:.1f} tok/s"
        )

        # Merge prompts + completions
        prompt_tokenized = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completions = pad(
            completion_ids_tensors, padding_value=self.processing_class.pad_token_id, padding_side="right"
        )

        reasoning_ids = torch.cat([prompt_ids, padded_completions], dim=1)

        return reasoning_ids

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # Recurse into child modules
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # Skip already visited FSDP subtrees
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = (
                            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        )
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Sync student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # For PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging,
            # because merging adapters in a sharded manner is not supported
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights during parameter gathering
                if self.is_fsdp_enabled:  # Note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights during parameter gathering
                    # For PEFT and FSDP, we need to use memory-efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to restore original parameter names and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When saving modules, remove prefix and discard original modules
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will be automatically repartitioned on context exit
        else:
            # For non-PEFT models, just gather (if needed) and update each parameter individually
            if self.is_fsdp_enabled:
                # Use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    def _save_generation_outputs(self, step: int):
        """Save generation outputs to disk."""
        if not self.accelerator.is_main_process:
            return

        if len(self._generation_outputs_buffer) == 0:
            return

        import json
        from pathlib import Path

        # Create generation directory in output directory
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        output_file = generations_dir / f"generations_step_{step}.json"

        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Saved {len(self._generation_outputs_buffer)} generation outputs to:")
        print(f"  {output_file}")
        print(f"{'='*80}\n")

        # Clear buffer after saving
        self._generation_outputs_buffer.clear()

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Execute self-distillation training step.

        If reason_first=True:
        1. Generate teacher reasoning for the solution
        2. Append reasoning to teacher prompt
        3. Generate completion from student prompt
        4. Compute JSD loss

        Otherwise:
        1. Generate completion from student prompt
        2. Construct full sequences with generated content for both student and teacher
        3. Compute JSD loss on generated tokens
        """
        on_policy = True

        # === Reasoning phase (if enabled) ===
        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASONING PHASE: Teacher analyzing solution...")
            print(f"{'='*80}\n")

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # Generate teacher reasoning
                teacher_reasoning_ids = self.generate_teacher_reasoning(
                    unwrapped_model,
                    inputs["teacher_reasoning_prompts"],
                    inputs.get("teacher_reasoning_attention_mask"),
                )

                # Decode reasoning
                reasoning_prompt_len = inputs["teacher_reasoning_prompt_length"]
                reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]
                reasoning_texts = self.processing_class.batch_decode(
                    reasoning_completions, skip_special_tokens=True
                )

                # Occasionally print reasoning
                if random.random() < 0.01:
                    print(f"\n{'='*80}")
                    print(f"TEACHER REASONING SAMPLE (Step {self.state.global_step}):")
                    print(f"{'='*80}")
                    sample_idx = random.randint(0, len(reasoning_texts) - 1)
                    print(f"\n{'='*80}")
                    # Decode prompts from token IDs to text
                    sample_prompt = self.processing_class.decode(
                        inputs["teacher_reasoning_prompts"][sample_idx], skip_special_tokens=False
                    )
                    print(f"PROMPT:\n{sample_prompt}")
                    print(f"\nReasoning:\n{reasoning_texts[sample_idx]}")
                    print(f"{'='*80}\n")

                # Update teacher prompts with reasoning
                # Construct: [teacher_reasoning_prompt][reasoning][transition_to_teaching]
                teacher_prompts_with_reasoning = torch.cat(
                    [
                        inputs["teacher_reasoning_prompts"],
                        reasoning_completions,
                        inputs["teacher_transition_tokens"],
                    ],
                    dim=1,
                )

                # Update inputs with new teacher prompts
                inputs["teacher_prompts"] = teacher_prompts_with_reasoning
                teacher_attention_mask = torch.ones_like(teacher_prompts_with_reasoning)
                if self.processing_class.pad_token_id is not None:
                    teacher_attention_mask[
                        teacher_prompts_with_reasoning == self.processing_class.pad_token_id
                    ] = 0
                inputs["teacher_prompt_attention_mask"] = teacher_attention_mask
                inputs["teacher_prompt_length"] = teacher_prompts_with_reasoning.shape[1]

        # === Generation phase ===
        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs, self.generation_config, self.processing_class.pad_token_id
            )
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
                generated_ids, generated_attention_mask, _ = result
                # Decode for logging
                prompt_texts = self.processing_class.batch_decode(
                    inputs["student_prompts"], skip_special_tokens=False
                )
                student_prompt_len = inputs["student_prompt_length"]
                completion_ids = generated_ids[:, student_prompt_len:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=False
                )

        # Get batch-level student prompt length
        student_prompt_len = inputs["student_prompt_length"]

        # Extract generation part (same slice for all examples because prompts are padded)
        generation_ids = generated_ids[:, student_prompt_len:]

        # Construct student full sequence: [student_prompt][generation]
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        # Construct teacher full sequence: [teacher_prompt][generation]
        teacher_prompts = inputs["teacher_prompts"]
        teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)

        # Create attention mask for teacher
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0

        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        # Create labels for generated tokens
        # Mask prompt tokens (use actual length per example for accurate masking)
        labels = generated_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
            labels[i, :actual_prompt_len] = -100  # Mask actual prompt

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        inputs["labels"] = labels

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        # Collect generation outputs to save
        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )

        # Occasionally print student generation with 1% probability
        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"STUDENT GENERATION SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nPrompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"{'='*80}\n")

        loss = super().training_step(model, inputs, num_items_in_batch)

        # Save generation outputs every N steps
        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # Average metrics

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            # Track on-policy/off-policy loss statistics
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Accumulate vector in distributed environment
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
            ) = vec.tolist()

            # Compute category averages, covering trainer's log window
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # Reset window accumulators after logging (like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

        # Process average of sequence-level KL divergences
        seq_kl_divergences = self._metrics[mode].get("seq_kl_divergences", [])
        if seq_kl_divergences:
            avg_seq_kl_div = sum(seq_kl_divergences) / len(seq_kl_divergences)
            metrics["seq_kl_divergence"] = avg_seq_kl_div

        # This method can be called in training and evaluation. When called in evaluation, keys in logs start with "eval_".
        # We need to add "eval_" prefix to keys in metrics to match format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        # Check logging system type and log accordingly
        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):
            # Determine logging system based on log_with parameter
            if self.log_with == "wandb":
                if self.args.report_to and "wandb" in self.args.report_to:
                    try:
                        import wandb
                        if wandb.run is not None:
                            import pandas as pd

                            table = {
                                "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                                "prompt": self._textual_logs["prompt"],
                                "completion": self._textual_logs["completion"],
                            }
                            df = pd.DataFrame(table)
                            if self.wandb_log_unique_prompts:
                                df = df.drop_duplicates(subset=["prompt"])
                            if self.num_completions_to_print and len(df) > 0:
                                df = df.sample(n=self.num_completions_to_print, random_state=42)
                            wandb.log({"completions": wandb.Table(dataframe=df)})
                    except ImportError:
                        pass  # wandb not installed
            elif self.log_with == "swanlab":
                try:
                    import swanlab
                    if swanlab.get_run() is not None:
                        import pandas as pd

                        table = {
                            "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                            "prompt": self._textual_logs["prompt"],
                            "completion": self._textual_logs["completion"],
                        }
                        df = pd.DataFrame(table)
                        if self.wandb_log_unique_prompts:  # Keep compatibility
                            df = df.drop_duplicates(subset=["prompt"])
                        if self.num_completions_to_print and len(df) > 0:
                            df = df.sample(n=self.num_completions_to_print, random_state=42)
                        
                        # Log to swanlab
                        swanlab.log({"completions": df.to_dict()})
                        
                        # Specially record sequence-level KL divergence
                        if seq_kl_divergences:
                            avg_seq_kl_div = sum(seq_kl_divergences) / len(seq_kl_divergences)
                            swanlab.log({"seq_kl_divergence": avg_seq_kl_div})
                except ImportError:
                    pass  # swanlab not installed

