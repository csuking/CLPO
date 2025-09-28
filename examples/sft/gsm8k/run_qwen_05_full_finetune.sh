#!/bin/bash
# Full parameter fine-tuning script for Qwen2.5-0.5B
# Tested with 4 GPUs (cards 4,5,6,7)

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_full_finetune.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Set CUDA_VISIBLE_DEVICES to use specific GPUs (4,5,6,7)
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/data/oss/xuexin.zsj/web_relevance_data/train_1.parquet \
    data.val_files=/mnt/data/oss/xuexin.zsj/web_relevance_data/train_1.parquet \
    data.prompt_key=prompt \
    data.response_key=prompt \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=web-relevance-full-finetune \
    trainer.experiment_name=web-relevance-full-finetune-qwen-2.5-0.5b \
    trainer.logger=console \
    trainer.total_epochs=1 $@ \
    model.lora_rank=0 \
    model.lora_alpha=0 \
    model.target_modules=all-linear

# Note: lora_rank=0 and lora_alpha=0 disables LoRA, enabling full parameter fine-tuning
# Note: Using prompt field directly since extra_info only contains {'split': 'train'}
