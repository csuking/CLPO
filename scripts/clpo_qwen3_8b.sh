
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}


# Data paths
train_file="path/to/train.json"
val_file="path/to/val.json"

# Model path
model_path="path/to/model"

# Output directory
output_dir="path/to/output"

# Data configuration
max_prompt_length=2048
max_response_length=8192
train_batch_size=64
val_batch_size=32
truncation="error"
filter_overlong_prompts=true
dataloader_num_workers=4

# CLPO ablation study configuration - BOTH
clpo_rewrite_mode="both"

# Algorithm configuration
adv_estimator=grpo
# Model configuration
enable_gradient_checkpointing=true
use_remove_padding=true

# Actor configuration
actor_lr=1e-6
actor_lr_warmup_steps=10
warmup_style=constant
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
use_kl_loss=true
kl_loss_coef_hard_scale=0.5
kl_loss_coef_nonhard_scale=1.0
kl_loss_coef=0.001
entropy_coeff=0
param_offload=false
optimizer_offload=false

# Rollout configuration
rollout_name=vllm
n_resp_per_prompt=4
tensor_model_parallel_size=1
gpu_memory_utilization=0.5
log_prob_micro_batch_size_per_gpu=1
max_model_len=10240
max_num_batched_tokens=10240

# Trainer configuration
total_epochs=1
critic_warmup=0
test_freq=10
save_freq=10
val_before_train=true
ngpus_per_node=${NGPUS_PER_NODE:-8}
nnodes=${NNODES:-1}

# CLPO specific configuration
clpo_hard_acc_upper=0.3
clpo_med_acc_lower=0.3
clpo_med_acc_upper=0.7

# CLPO rewrite data saving configuration
clpo_save_rewrite_data=true
clpo_rewrite_save_path="path/to/rewrite_data.json"
clpo_hard_rewrite_save_path="path/to/hard_rewrite_data.json"
clpo_medium_rewrite_save_path="path/to/medium_rewrite_data.json"

echo "Train Data: $train_file"
echo "Val Data: $val_file"
echo "Model Path: $model_path"
echo "Output Dir: $output_dir"
echo "CLPO Hard Acc Upper: $clpo_hard_acc_upper"
echo "CLPO Med Acc Lower: $clpo_med_acc_lower"
echo "CLPO Med Acc Upper: $clpo_med_acc_upper"
echo "CLPO Save Rewrite Data: $clpo_save_rewrite_data"
echo "CLPO Rewrite Save Path: $clpo_rewrite_save_path"
echo "CLPO Hard Rewrite Save Path: $clpo_hard_rewrite_save_path"
echo "CLPO Medium Rewrite Save Path: $clpo_medium_rewrite_save_path"


python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${adv_estimator}" \
  algorithm.norm_adv_by_std_in_grpo=true \
  data.train_files="${train_file}" \
  data.val_files="${val_file}" \
  data.train_batch_size="${train_batch_size}" \
  data.val_batch_size="${val_batch_size}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts="${filter_overlong_prompts}" \
  data.truncation="${truncation}" \
  data.dataloader_num_workers="${dataloader_num_workers}" \
  data.clpo_hard_acc_upper="${clpo_hard_acc_upper}" \
  data.clpo_rewrite_mode="${clpo_rewrite_mode}" \
  data.clpo_medium_acc_lower="${clpo_med_acc_lower}" \
  data.clpo_medium_acc_upper="${clpo_med_acc_upper}" \
  data.clpo_save_rewrite_data="${clpo_save_rewrite_data}" \
  data.clpo_rewrite_save_path="${clpo_rewrite_save_path}" \
  data.clpo_hard_rewrite_save_path="${clpo_hard_rewrite_save_path}" \
  data.clpo_medium_rewrite_save_path="${clpo_medium_rewrite_save_path}" \
  actor_rollout_ref.model.path="${model_path}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${enable_gradient_checkpointing}" \
  actor_rollout_ref.model.use_remove_padding="${use_remove_padding}" \
  actor_rollout_ref.actor.optim.lr="${actor_lr}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${actor_lr_warmup_steps}" \
  actor_rollout_ref.actor.optim.warmup_style="${warmup_style}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
  actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
  actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
  actor_rollout_ref.actor.kl_loss_coef_hard_scale="${kl_loss_coef_hard_scale}" \
  actor_rollout_ref.actor.kl_loss_coef_nonhard_scale="${kl_loss_coef_nonhard_scale}" \
  actor_rollout_ref.actor.entropy_coeff="${entropy_coeff}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${param_offload}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${optimizer_offload}" \
  actor_rollout_ref.rollout.name="${rollout_name}" \
  actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${tensor_model_parallel_size}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${log_prob_micro_batch_size_per_gpu}" \
  actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${max_num_batched_tokens}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${log_prob_micro_batch_size_per_gpu}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${param_offload}" \
  trainer.logger='["console"]' \
  trainer.total_epochs="${total_epochs}" \
  trainer.critic_warmup="${critic_warmup}" \
  trainer.test_freq="${test_freq}" \
  trainer.save_freq="${save_freq}" \
  trainer.val_before_train="${val_before_train}" \
  trainer.n_gpus_per_node="${ngpus_per_node}" \
  trainer.nnodes="${nnodes}" \
  trainer.default_local_dir="${output_dir}" \
  trainer.task=clpo \
  "$@"
