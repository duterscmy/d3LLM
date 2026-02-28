import sys
import os

sys.path.append("./distill_2_training")
import torch
import torch.nn.functional as F
import yaml
from datasets import load_from_disk, load_dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from typing import Dict, Any, List
from dataclasses import dataclass
import random
import pickle
import hashlib
import os
import subprocess
from ast import literal_eval


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Override config values from command line args like 'training.learning_rate=0.000001'"""
    for override in overrides:
        # Skip args (--local_rank, etc.)
        if override.startswith('--') or '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to nested dict
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        
        # Convert value to appropriate type
        final_key = keys[-1]
        old_value = target.get(final_key)
        
        # Try literal_eval for lists/dicts (e.g., "[16,32,32]")
        if isinstance(old_value, (list, dict)) or value.startswith(('[', '{')):
            try:
                target[final_key] = literal_eval(value)
            except (ValueError, SyntaxError):
                target[final_key] = value
        elif isinstance(old_value, bool):
            target[final_key] = value.lower() in ('true', '1', 'yes')
        elif isinstance(old_value, int):
            target[final_key] = int(value)
        elif isinstance(old_value, float):
            target[final_key] = float(value)
        else:
            target[final_key] = value
    
    return config


def prepare_model(config: Dict[str, Any]):
    """Prepare model and tokenizer with LoRA"""
    torch_dtype = getattr(torch, config["model"]["torch_dtype"])
    
    model = AutoModel.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch_dtype,
        trust_remote_code=config["model"]["trust_remote_code"],
        device_map="auto",  # 自动分配到GPU
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], 
        trust_remote_code=config["model"]["trust_remote_code"]
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA if enabled in config
    lora_config_dict = config.get("lora")
    if lora_config_dict and lora_config_dict.get("enabled", False):
        print("=" * 80)
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 16),
            lora_alpha=lora_config_dict.get("lora_alpha", 16),
            target_modules=lora_config_dict.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=lora_config_dict.get("lora_dropout", 0.0),
            bias=lora_config_dict.get("bias", "none"),
            task_type=lora_config_dict.get("task_type", "CAUSAL_LM")
        )
        
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads() 
        model.config.use_cache = False
        
        # Print the number of trainable parameters
        model.print_trainable_parameters()
        print("=" * 80)
    else:
        print("=" * 80)
        print("LoRA is disabled. Training full model.")
        print("=" * 80)
    
    return model, tokenizer


def select_trajectory_by_ratio(trajectories, mask_ratio, mask_token_id, block_start, block_end):
    """Select the trajectory step with mask ratio closest to target mask ratio in the current block"""
    if not trajectories or len(trajectories) == 0:
        return None
    
    # Trajectories are ordered by denoising steps (from fully masked to clean)
    # Calculate the target number of unmasked tokens in the current block
    block_length = block_end - block_start
    num_unmasked = int((1 - mask_ratio) * block_length)
    
    # Map to trajectory index: earlier steps have more masks
    # index = block_start + number of unmasked tokens so far
    target_idx = block_start + num_unmasked
    target_idx = min(target_idx, len(trajectories) - 1)

    print("len(trajectories):{}".format(len(trajectories)))
    print("len(trajectories[target_idx]):{}".format(len(trajectories[target_idx])))
    
    return trajectories[target_idx]


def naive_random_mask(trajectories, mask_ratio, mask_token_id, block_start, block_end):
    """Baseline: randomly mask final trajectory by mask_ratio in specified block region"""
    return None


def forward_process_with_trajectory(
    input_ids,
    prompt_lengths,
    trajectory_batch,
    mask_token_id=151666,
    block_size=32,
    mask_ratio=0.5,
    use_blockwise=False,
    use_naive_random_mask=False,
    use_complementary_loss=False,
    eps=1e-3,
):
    """Forward masking using teacher trajectories"""
    b, l = input_ids.shape
    device = input_ids.device
    
    print("input ids size:{}".format(input_ids.size()))
    noisy_batch = input_ids.clone()
    noisy_batch_rev = input_ids.clone() if use_complementary_loss else None
    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    masked_indices_rev = torch.zeros_like(input_ids, dtype=torch.bool) if use_complementary_loss else None
    print("masked_indices size:{}".format(masked_indices.size()))
    # Protect prompt region from masking
    token_positions = torch.arange(l, device=device).expand(b, l)
    prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
    
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    if use_complementary_loss:
        noisy_batch_rev[prompt_mask] = input_ids[prompt_mask]
    
    for i in range(b):
        prompt_len = prompt_lengths[i].item()
        response_len = l - prompt_len
        print("prompt len, response len: {} {}".format(prompt_len, response_len))
        if response_len <= 0:
            continue
        
        # Determine mask region
        if use_blockwise:
            max_blocks = response_len // block_size
            # 确保 max_blocks 至少为 0
            max_blocks = max(0, max_blocks)
            num_blocks = random.randint(0, max_blocks)
            mask_start = prompt_len + num_blocks * block_size
            # mask_start = num_blocks * block_size
            # 确保 mask_end 不超过序列长度
            if num_blocks < max_blocks:
                mask_end = mask_start + block_size
            else:
                mask_end = l
        else:
            mask_start = prompt_len
            mask_end = l
        
        print("mask_start, mask_end: {}, {}".format(mask_start, mask_end))
        # 检查 mask region 是否有效
        if mask_start >= mask_end:
            # 如果没有有效区域，跳过这个样本的masking
            # 但仍然需要处理 future tokens
            noisy_batch[i, prompt_len:l] = mask_token_id
            if use_complementary_loss:
                noisy_batch_rev[i, prompt_len:l] = mask_token_id
            continue
        
        # Get trajectory or use random masking
        traj_fn = naive_random_mask if use_naive_random_mask else select_trajectory_by_ratio
        traj_step = traj_fn(
            trajectory_batch[i], mask_ratio, mask_token_id, mask_start - prompt_len, mask_end - prompt_len
        )
        
        # Extract or generate seg_mask
        seg_len = mask_end - mask_start
        if seg_len > 0:
            if traj_step is not None:
                traj_tensor = torch.tensor(traj_step, device=device, dtype=torch.long)
                print("traj_step.size(): {}".format(len(traj_step)))
                mask_start_traject, mask_end_traject = mask_start - prompt_len, mask_end - prompt_len
                print("mask_start_traject, mask_end_traject: {}, {}".format(mask_start_traject, mask_end_traject))
                seg_mask = (traj_tensor[mask_start_traject:mask_end_traject] == mask_token_id)
            else:
                p_mask = (1 - eps) * mask_ratio + eps
                seg_mask = torch.rand(seg_len, device=device) < p_mask
            
            # Apply mask
            print("seg_mask.size(): {}".format(seg_mask.size()))
            masked_indices[i, mask_start:mask_end] = seg_mask
            if use_complementary_loss:
                masked_indices_rev[i, mask_start:mask_end] = ~seg_mask
            
            noisy_batch[i, mask_start:mask_end] = torch.where(
                masked_indices[i, mask_start:mask_end], mask_token_id, input_ids[i, mask_start:mask_end]
            )
            if use_complementary_loss:
                noisy_batch_rev[i, mask_start:mask_end] = torch.where(
                    masked_indices_rev[i, mask_start:mask_end], mask_token_id, input_ids[i, mask_start:mask_end]
                )
        
        # Mask future tokens (always do this regardless of seg_len)
        if mask_end < l:
            noisy_batch[i, mask_end:l] = mask_token_id
            if use_complementary_loss:
                noisy_batch_rev[i, mask_end:l] = mask_token_id

    if use_complementary_loss:
        return noisy_batch, noisy_batch_rev, masked_indices, masked_indices_rev
    return noisy_batch, masked_indices


class DLMTrainer(Trainer):
    """Trajectory-based diffusion language model trainer for DREAM"""
    
    def __init__(
        self,
        mask_token_id=151666,
        temperature=0.5,
        entropy_weight=1.0,
        progressive_block_sizes=None,
        min_mask_ratio=0.2,
        max_mask_ratio=0.8,
        use_blockwise_loss=False,
        use_naive_random_mask=False,
        use_complementary_loss=False,
        trajectory_dataset=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.progressive_block_sizes = progressive_block_sizes or [32]
        self.current_block_size = self.progressive_block_sizes[0]
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.use_blockwise_loss = use_blockwise_loss
        self.use_naive_random_mask = use_naive_random_mask
        self.use_complementary_loss = use_complementary_loss
        self.trajectory_dataset = trajectory_dataset
    
    def get_current_block_size(self):
        """Calculate current block size based on epoch progress (linear interpolation)"""
        if self.state.epoch is None:
            return self.progressive_block_sizes[0]
        
        current_epoch = self.state.epoch
        epoch_idx = int(current_epoch)
        epoch_idx = min(epoch_idx, len(self.progressive_block_sizes) - 1)
        
        start_block_size = self.progressive_block_sizes[epoch_idx]
        
        if epoch_idx >= len(self.progressive_block_sizes) - 1:
            return int(start_block_size)
        
        end_block_size = self.progressive_block_sizes[epoch_idx + 1]
        epoch_progress = current_epoch - epoch_idx
        interpolated_size = start_block_size + epoch_progress * (end_block_size - start_block_size)
        
        return int(interpolated_size)
    
    def get_current_mask_ratio(self):
        """Calculate current mask ratio based on training progress (linear schedule)"""
        if self.state.max_steps > 0:
            current_step = self.state.global_step
            total_steps = self.state.max_steps
            progress = min(current_step / total_steps, 1.0)
            current_ratio = self.min_mask_ratio + progress * (self.max_mask_ratio - self.min_mask_ratio)
            return current_ratio
        else:
            return self.min_mask_ratio
    
    def _get_gpu_stats(self):
        """Get GPU memory and utilization statistics"""
        if torch.cuda.is_available():
            return {
                'gpu_memory_used_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_total_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
                'gpu_utilization_percent': 0,  # Can't get utilization easily without nvidia-smi
                'num_gpus': 1
            }
        return None
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Override log to add GPU statistics and current mask ratio"""
        gpu_stats = self._get_gpu_stats()
        if gpu_stats:
            logs.update(gpu_stats)
        
        logs['mask_ratio'] = self.get_current_mask_ratio()
        logs['block_size'] = self.get_current_block_size()
        
        super().log(logs, *args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs["prompt_lengths"]
        sample_indices = inputs["sample_idx"]
        
        # 1. 加载 Trajectories
        trajectories = []
        for idx in sample_indices.cpu().tolist():
            if self.trajectory_dataset is not None and idx < len(self.trajectory_dataset):
                traj = self.trajectory_dataset[idx]["trajectory"]
            else:
                traj = []
            trajectories.append(traj)
        
        # 2. 获取当前 Mask 策略
        current_mask_ratio = self.get_current_mask_ratio()
        current_mask_ratio = random.uniform(current_mask_ratio, self.max_mask_ratio)
        current_block_size = self.get_current_block_size()
        
        # 3. 生成 Mask (Forward Process)
        if self.use_complementary_loss:
            noisy_batch, noisy_batch_rev, masked_indices, masked_indices_rev = forward_process_with_trajectory(
                input_ids, prompt_lengths, trajectories,
                mask_token_id=self.mask_token_id, block_size=current_block_size,
                mask_ratio=current_mask_ratio, use_blockwise=self.use_blockwise_loss,
                use_naive_random_mask=self.use_naive_random_mask,
                use_complementary_loss=True,
            )
        else:
            noisy_batch, masked_indices = forward_process_with_trajectory(
                input_ids, prompt_lengths, trajectories,
                mask_token_id=self.mask_token_id, block_size=current_block_size,
                mask_ratio=current_mask_ratio, use_blockwise=self.use_blockwise_loss,
                use_naive_random_mask=self.use_naive_random_mask,
            )
        
        # 4. 前向传播
        outputs = model(input_ids=noisy_batch)
        logits = outputs.logits[:, :-1].float()

        if self.use_complementary_loss:
            outputs_rev = model(input_ids=noisy_batch_rev)
            logits_rev = outputs_rev.logits[:, :-1].float()
        
        # 准备 Label 和 Mask
        input_ids_shifted = input_ids[:, 1:]
        masked_indices_shifted = masked_indices[:, 1:]
        
        # 5. 计算主路径 CE Loss
        if masked_indices_shifted.any():
            ce_loss = F.cross_entropy(logits[masked_indices_shifted], input_ids_shifted[masked_indices_shifted])
        else:
            ce_loss = torch.tensor(0.0, device=input_ids.device)

        # 6. 计算互补路径 CE Loss
        ce_loss_rev = torch.tensor(0.0, device=input_ids.device)
        if self.use_complementary_loss:
            masked_indices_rev_shifted = masked_indices_rev[:, 1:]
            if masked_indices_rev_shifted.any():
                ce_loss_rev = F.cross_entropy(logits_rev[masked_indices_rev_shifted], input_ids_shifted[masked_indices_rev_shifted])

        # 7. 计算主路径 Entropy Loss
        entropy_loss = torch.tensor(0.0, device=input_ids.device)
        if masked_indices_shifted.any():
            probs = F.softmax(logits / self.temperature, dim=-1)
            H_tok = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
            pred_ids = logits.argmax(dim=-1)
            correct_mask = (pred_ids == input_ids_shifted) & masked_indices_shifted
            num_correct = correct_mask.sum()
            if num_correct > 0:
                entropy_loss = (H_tok * correct_mask).sum() / num_correct.clamp_min(1)

        # 8. 计算互补路径 Entropy Loss
        entropy_loss_rev = torch.tensor(0.0, device=input_ids.device)
        if self.use_complementary_loss:
            masked_indices_rev_shifted = masked_indices_rev[:, 1:]
            if masked_indices_rev_shifted.any():
                probs_rev = F.softmax(logits_rev / self.temperature, dim=-1)
                H_tok_rev = -(probs_rev * torch.log(probs_rev + 1e-12)).sum(dim=-1)
                pred_ids_rev = logits_rev.argmax(dim=-1)
                correct_mask_rev = (pred_ids_rev == input_ids_shifted) & masked_indices_rev_shifted
                num_correct_rev = correct_mask_rev.sum()
                if num_correct_rev > 0:
                    entropy_loss_rev = (H_tok_rev * correct_mask_rev).sum() / num_correct_rev.clamp_min(1)

        # 9. 合并 Loss
        if self.use_complementary_loss:
            total_loss = (ce_loss + ce_loss_rev + self.entropy_weight * (entropy_loss + entropy_loss_rev)) / 4.0
        else:
            total_loss = (ce_loss + self.entropy_weight * entropy_loss)
        
        print(f"LOSS: {total_loss.item()}")
        return (total_loss, outputs) if return_outputs else total_loss
        


def main():
    # 1. Load configuration, model and tokenizer
    import os
    from datetime import datetime
    import shutil
    from zoneinfo import ZoneInfo
    
    config_path = os.path.join(os.path.dirname(__file__), "d3llm_train.yaml")
    config = load_config(config_path)
    
    # Override config from command line args
    config = override_config(config, sys.argv[1:])
    
    # Save modified config as d3llm_train_used.yaml for backup
    used_config_path = os.path.join(os.path.dirname(__file__), "d3llm_train_used.yaml")
    with open(used_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Print all configuration parameters
    print(f"=" * 80)
    print("Configuration Parameters:")
    print(f"=" * 80)
    import json
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"=" * 80)
    
    # Get SLURM job ID if available
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    
    # Use timestamp from environment (set by shell) or generate new one
    timestamp = os.environ.get("TRAINING_TIMESTAMP")
    if not timestamp:
        san_diego_tz = ZoneInfo("America/Los_Angeles")
        timestamp = datetime.now(san_diego_tz).strftime("%m%d_%H%M%S")
    
    base_output_dir = config["training"]["output_dir"]
    output_dir = f"{base_output_dir}_{slurm_job_id}_{timestamp}"
    
    # Create W&B run name with the same format
    wandb_run_name = f"{os.path.basename(base_output_dir)}_{slurm_job_id}_{timestamp}"
    
    # Update config with timestamped output_dir and run_name
    config["training"]["output_dir"] = output_dir
    config["training"]["run_name"] = wandb_run_name
    
    print(f"=" * 80)
    print(f"SLURM Job ID: {slurm_job_id}")
    print(f"Output directory: {output_dir}")
    print(f"W&B Run name: {wandb_run_name}")
    print(f"=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Backup training code and config to output directory
    source_dir = os.path.dirname(__file__)
    backup_dir = os.path.join(output_dir, "training_code_backup")
    
    print(f"Backing up training code from {source_dir} to {backup_dir}...")
    
    try:
        shutil.copytree(source_dir, backup_dir, dirs_exist_ok=True)
        print(f"Training code backed up successfully!")
    except Exception as e:
        print(f"Warning: Failed to backup training code: {e}")
        print(f"Continuing with training anyway...")
    
    print(f"=" * 80)
    
    # 移除 DeepSpeed 配置，添加单卡训练参数
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        overwrite_output_dir=config["training"].get("overwrite_output_dir", True),
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
        warmup_steps=config["training"].get("warmup_steps", 0),
        logging_steps=config["training"].get("logging_steps", 10),
        save_steps=config["training"].get("save_steps", 500),
        eval_steps=config["training"].get("eval_steps", None),
        save_total_limit=config["training"].get("save_total_limit", 3),
        load_best_model_at_end=config["training"].get("load_best_model_at_end", False),
        report_to=config["training"].get("report_to", "wandb"),
        run_name=config["training"]["run_name"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        label_names=["input_ids", "prompt_lengths", "sample_idx"],
        # 单卡训练不需要 deepspeed 和分布式相关参数
        local_rank=-1,
        ddp_backend=None,
    )
    
    print(f"Training config: single GPU mode")
    print(f"Is DeepSpeed enabled: No")
    
    model, tokenizer = prepare_model(config)
    
    # 2. Load trajectory dataset and create mapping
    distill_config = config.get("distillation", {})
    trajectory_dataset_path = distill_config.get("trajectory_dataset_path")
    
    if trajectory_dataset_path:
        # Handle relative path from script directory (only if it's a local path)
        if not os.path.isabs(trajectory_dataset_path):
            potential_path = os.path.join(os.path.dirname(__file__), "..", trajectory_dataset_path)
            if os.path.exists(potential_path):
                trajectory_dataset_path = potential_path
        
        # Create cache directory and file path
        cache_dir = os.path.join(os.path.dirname(trajectory_dataset_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache key based on dataset path and max_length (important for preprocessing)
        max_length = distill_config.get("max_length", 512)
        cache_params = f"{trajectory_dataset_path}_maxlen{max_length}".encode()
        cache_key = hashlib.md5(cache_params).hexdigest()
        cache_file = os.path.join(cache_dir, f"trajectory_preprocessed_{cache_key}.pkl")
        
        # Try to load preprocessed trajectory dataset from cache first
        if os.path.exists(cache_file):
            try:
                print(f"Loading preprocessed trajectory dataset from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    trajectory_dataset = pickle.load(f)
                print(f"Successfully loaded {len(trajectory_dataset)} preprocessed samples from cache!")
            except Exception as e:
                print(f"Failed to load cache: {e}")
                print(f"Will process dataset from scratch...")
                trajectory_dataset = None
        else:
            print(f"Preprocessed cache not found at {cache_file}. Processing from scratch...")
            trajectory_dataset = None
        
        # If cache doesn't exist or failed to load, process dataset
        if trajectory_dataset is None:
            print(f"Loading trajectory dataset from {trajectory_dataset_path}...")
            # Support both local and remote HuggingFace datasets
            if os.path.isdir(trajectory_dataset_path):
                trajectory_dataset = load_from_disk(trajectory_dataset_path)
            else:
                trajectory_dataset = load_dataset(trajectory_dataset_path, split="train")
            
            # Filter correct samples only
            num_proc = distill_config.get("num_proc", 8)
            print(f"Filtering correct trajectory samples with {num_proc} processes...")
            trajectory_dataset = trajectory_dataset.filter(lambda x: x["is_correct"], num_proc=num_proc)
            print(f"Loaded {len(trajectory_dataset)} correct trajectory samples")
            
            # Preprocess trajectory dataset: truncate and pad each step to max_length
            pad_token_id = tokenizer.eos_token_id
            
            def preprocess_trajectory_sample(examples):
                """Preprocess trajectory samples: truncate and pad each step to max_length"""
                processed_trajectories = []
                
                for traj in examples["trajectory"]:
                    if traj:
                        padded_traj = []
                        for step in traj:
                            if len(step) < max_length:
                                # Pad with eos_token to max_length
                                padding_length = max_length - len(step)
                                padded_step = step + [pad_token_id] * padding_length
                            else:
                                # Truncate to max_length
                                padded_step = step[:max_length]
                            padded_traj.append(padded_step)
                        processed_trajectories.append(padded_traj)
                    else:
                        processed_trajectories.append([])
                
                return {
                    "trajectory": processed_trajectories,
                }
            
            print(f"Preprocessing trajectories (truncate/pad to max_length={max_length})...")
            trajectory_dataset = trajectory_dataset.map(
                preprocess_trajectory_sample,
                batched=True,
                num_proc=num_proc,
                desc="Preprocessing trajectories"
            )
            print(f"Trajectory preprocessing completed!")
            
            # Save preprocessed dataset to cache
            try:
                print(f"Saving preprocessed trajectory dataset to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(trajectory_dataset, f)
                print(f"Preprocessed cache saved successfully!")
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")
        
        print(f"Preprocessed trajectory dataset ready with {len(trajectory_dataset)} samples")
    else:
        print(f"No trajectory dataset specified, using random masking")
        trajectory_dataset = None
    
    # 3. Load the original dataset
    dataset = trajectory_dataset

    # Limit dataset size for testing if max_samples is specified
    max_samples = distill_config.get("max_samples")
    if max_samples is not None and max_samples > 0:
        original_size = len(dataset)
        dataset = dataset.select(range(min(max_samples, original_size)))
        print(f"=" * 80)
        print(f"[Testing Mode] Limited dataset from {original_size} to {len(dataset)} samples")
        
        # Also limit trajectory dataset to match
        if trajectory_dataset is not None:
            traj_original_size = len(trajectory_dataset)
            trajectory_dataset = trajectory_dataset.select(range(min(max_samples, traj_original_size)))
            print(f"[Testing Mode] Limited trajectory dataset from {traj_original_size} to {len(trajectory_dataset)} samples")
        print(f"=" * 80)
    
    # 4. Check tokenized dataset cache
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on dataset configuration
    cache_params = {
        "model_name": config["model"]["name"],
        "max_samples": max_samples,
        "max_length": distill_config.get("max_length", 512),
        "dataset_size": len(dataset),
    }
    cache_key_str = str(cache_params).encode()
    cache_key = hashlib.md5(cache_key_str).hexdigest()
    cache_file_tokenized = os.path.join(cache_dir, f"tokenized_dataset_{cache_key}.pkl")
    
    tokenized_dataset = None
    # Try to load tokenized dataset from cache
    # if os.path.exists(cache_file_tokenized):
    #     try:
    #         print(f"=" * 80)
    #         print(f"Loading tokenized dataset from cache: {cache_file_tokenized}")
    #         with open(cache_file_tokenized, 'rb') as f:
    #             tokenized_dataset = pickle.load(f)
    #         print(f"Successfully loaded tokenized dataset with {len(tokenized_dataset)} samples from cache!")
    #         print(f"=" * 80)
    #     except Exception as e:
    #         print(f"Failed to load tokenized dataset cache: {e}")
    #         print(f"Will tokenize from scratch...")
    #         tokenized_dataset = None
    # else:
    #     print(f"Tokenized dataset cache not found. Will tokenize from scratch...")
    #     tokenized_dataset = None
    
    # If cache doesn't exist or failed to load, perform tokenization
    if tokenized_dataset is None:
        # Format each sample, generate the complete text and record the number of tokens in the prompt section
        def format_example(example):
            texts = []
            prompt_lengths = []
            idx = 0
            for question, response in zip(example["question"], example["generated_text"]):
                # prompt text
                real_question = question.split('\n')[0]
                real_response = response.split('dont repeat question:<|im_end|>\n<|im_start|>assistant\n')[-1]
                messages = [{"role": "user", "content": real_question}]
                prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                # response text
                answer_text = real_response + tokenizer.eos_token
                
                # complete text
                full_text = prompt_text + answer_text
                texts.append(full_text)
                
                if idx < 5:
                    print(f"=========sample {idx}==========")
                    print(real_question)
                    print(real_response)
                idx += 1

                # Calculate the number of tokens in the prompt part
                prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                prompt_lengths.append(len(prompt_token_ids))
            
            return {"text": texts, "prompt_length": prompt_lengths}
        
        print(f"Formatting dataset...")
        formatted_dataset = dataset.map(
            format_example,
            batched=True,
        )
        
        def tokenize_function(examples, indices):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=distill_config.get("max_length", 512),
                add_special_tokens=False,
            )
            
            tokenized["prompt_lengths"] = examples["prompt_length"]
            
            # Store original dataset index for dynamic trajectory loading during training
            tokenized["sample_idx"] = list(indices)
            
            return tokenized
        
        print(f"Tokenizing dataset...")
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            with_indices=True,
        )
        
        # Save tokenized dataset to cache
        try:
            print(f"Saving tokenized dataset to cache: {cache_file_tokenized}")
            with open(cache_file_tokenized, 'wb') as f:
                pickle.dump(tokenized_dataset, f)
            print(f"Tokenized dataset cache saved successfully!")
        except Exception as e:
            print(f"Warning: Failed to save tokenized dataset cache: {e}")
    
    from dataclasses import dataclass
    from typing import Dict, List, Any
    
    @dataclass
    class MaskDiffusionDataCollator:
        tokenizer: Any
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_ids = [torch.tensor(f["input_ids"]) for f in features]
            prompt_lengths = [f["prompt_lengths"] for f in features]
            sample_indices = [f["sample_idx"] for f in features]
            
            target_length = 256 + max(prompt_lengths)
            
            pad_token_id = self.tokenizer.eos_token_id
            
            # right padding
            padded_input_ids = []
            for ids in input_ids:
                current_length = len(ids)
                # print(f"[Debug-1] current_length: {current_length}, target_length: {target_length}")
                if current_length < target_length:
                    # Right padding with EOS token
                    padding_length = target_length - current_length
                    padded_ids = torch.cat([
                        ids,
                        torch.full((padding_length,), pad_token_id, dtype=ids.dtype)
                    ])
                else:
                    # Truncate to target_length
                    padded_ids = ids[:target_length]
                
                padded_input_ids.append(padded_ids)
            
            batch = {
                "input_ids": torch.stack(padded_input_ids),
                "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
                "sample_idx": torch.tensor(sample_indices, dtype=torch.long),
            }
            
            return batch
    
    data_collator_fixed = MaskDiffusionDataCollator(
        tokenizer=tokenizer,
    )
    
    # 5. Create trainer and train
    progressive_block_sizes = distill_config.get("progressive_block_sizes", [32])
    num_epochs = config["training"]["num_train_epochs"]
    
    # Validate progressive_block_sizes length
    if len(progressive_block_sizes) != num_epochs:
        print(f"Warning: progressive_block_sizes length ({len(progressive_block_sizes)}) != num_epochs ({num_epochs})")
        print(f"Using last block size ({progressive_block_sizes[-1]}) for remaining epochs")
        progressive_block_sizes = progressive_block_sizes + [
            progressive_block_sizes[-1]
        ] * (num_epochs - len(progressive_block_sizes))
    
    # 6. DLM Trainer
    model.config.use_cache = False

    trainer = DLMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator_fixed,
        mask_token_id=151666,
        temperature=distill_config.get("temperature", 0.5),
        entropy_weight=distill_config.get("entropy_weight", 1.0),
        progressive_block_sizes=progressive_block_sizes,
        min_mask_ratio=distill_config.get("min_mask_ratio", 0.2),
        max_mask_ratio=distill_config.get("max_mask_ratio", 0.8),
        use_blockwise_loss=distill_config.get("use_blockwise_loss", False),
        use_naive_random_mask=distill_config.get("use_naive_random_mask", False),
        use_complementary_loss=distill_config.get("use_complementary_loss", False),
        trajectory_dataset=trajectory_dataset,  # Pass trajectory_dataset for dynamic loading
    )
    
    print(f"Training with progressive block sizes: {trainer.progressive_block_sizes}")
    print(f"Starting with block size: {trainer.current_block_size}")
    print(f"Progressive mask ratio: [{trainer.min_mask_ratio}, {trainer.max_mask_ratio}]")
    print(f"Temperature: {trainer.temperature}, Entropy weight: {trainer.entropy_weight}")
    
    # 6. start training
    trainer.train()


if __name__ == "__main__":
    main()