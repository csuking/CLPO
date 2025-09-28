#!/usr/bin/env bash
# DAPO Multi-Node Training Script
set -euo pipefail

# ========================
# Environment and system
# ========================
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export SWANLAB_API_KEY=0YpQmYtrGLl6VHSFU0vis
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# ========================
# Multi-node configuration
# ========================
echo "==== MULTI-NODE CONFIGURATION ===="
echo "HOSTNAME=$(hostname)"
POD_IP=$(hostname -I | awk '{print $1}' || hostname -i)
echo "POD_IP=$POD_IP"
echo "RANK=${RANK:-0} WORLD_SIZE=${WORLD_SIZE:-1} MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}"

# GPU detection
nvidia-smi -L || true
NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
NNODES=${WORLD_SIZE:-1}
HEAD_SVC=${MASTER_ADDR:-127.0.0.1}

# ========================
# Project and experiment configuration
# ========================
project_name='AIME2024-Qwen3-8B'
timestamp=$(date +%Y%m%d_%H%M%S)
experiment_name="aime2024_dapo_multi_node_${timestamp}"

# Data paths
train_file="/primus_datasets/primus_data/clpo_SKYRTP/DAPO-Math-17k/data/dapo-math-17k.parquet"
val_file="/primus_datasets/primus_data/aime_2B4pCq/train-00000-of-00001-fixed.parquet"

# Model path
model_path="/primus_datasets/primus_data/Qwen3_rNrLUi/Qwen3-8B"

# Output directory
output_dir="/primus_oss/xuexin_checkpoint/0915-Qwen3-8B-AIME2024/dapo_multi_node"

# ========================
# Data configuration
# ========================
max_prompt_length=2048
max_response_length=8192
gen_batch_size=192
train_batch_size=64
val_batch_size=32
truncation="error"
filter_overlong_prompts=true
dataloader_num_workers=4

# ========================
# Algorithm configuration
# ========================
adv_estimator=grpo
use_kl_in_reward=false
filter_groups_enable=True
filter_groups_metric=acc
max_num_gen_batches=3

# ========================
# Model configuration
# ========================
enable_gradient_checkpointing=true
use_remove_padding=true

# ========================
# Actor configuration
# ========================
actor_lr=1e-6
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
use_kl_loss=false
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
entropy_coeff=0
param_offload=false
optimizer_offload=false

# ========================
# Rollout configuration
# ========================
rollout_name=vllm
n_resp_per_prompt=8
tensor_model_parallel_size=1
gpu_memory_utilization=0.5
log_prob_micro_batch_size_per_gpu=1
max_model_len=10240
max_num_batched_tokens=10240

# ========================
# DAPO specific configuration
# ========================
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# ========================
# Trainer configuration
# ========================
total_epochs=1
critic_warmup=0
test_freq=1
save_freq=10
val_before_train=true
ngpus_per_node=${NGPUS_PER_NODE:-8}
nnodes=${NNODES:-1}

# ========================
# Validation parameters
# ========================
val_rollout_n=32
val_do_sample=true
val_temperature=1.0
val_top_k=-1
val_top_p=0.7

# ========================
# Ray cluster health check functions
# ========================
check_ray_ready() {
  local max_attempts=30
  local attempt=1
  
  echo "🔍 Checking Ray cluster status..."
  while [ $attempt -le $max_attempts ]; do
    if ray status >/dev/null 2>&1; then
      echo "✅ Ray cluster is ready!"
      ray status
      return 0
    fi
    echo "⏳ Attempt $attempt/$max_attempts: Ray not ready, waiting..."
    sleep 10
    ((attempt++))
  done
  
  echo "❌ Ray cluster failed to start after $max_attempts attempts"
  return 1
}

wait_for_master() {
  local max_attempts=60
  local attempt=1
  
  echo "🔗 Waiting for master Ray service at $HEAD_SVC:6379..."
  while [ $attempt -le $max_attempts ]; do
    if timeout 5 bash -c "</dev/tcp/$HEAD_SVC/6379" 2>/dev/null; then
      echo "✅ Master Ray service is accessible!"
      return 0
    fi
    echo "⏳ Attempt $attempt/$max_attempts: Master not ready, waiting..."
    sleep 10
    ((attempt++))
  done
  
  echo "❌ Master Ray service not accessible after $max_attempts attempts"
  return 1
}

# ========================
# Main execution logic
# ========================
if [[ "${RANK:-0}" == "0" ]]; then
  echo "🎯 == Starting Ray HEAD Node =="
  echo "Project: $project_name"
  echo "Experiment: $experiment_name"
  echo "Train Data: $train_file"
  echo "Val Data: $val_file"
  echo "Model Path: $model_path"
  echo "Output Dir: $output_dir"
  echo "Nodes: $NNODES, GPUs per node: $ngpus_per_node"
  echo "DAPO Overlong Buffer: $enable_overlong_buffer"
  echo "Train n_resp_per_prompt: $n_resp_per_prompt"
  
  # Start Ray head node
  ray start --head --node-ip-address "$POD_IP" --port 6379 \
    --dashboard-host 0.0.0.0 --dashboard-port 8265 \
    --num-gpus "$NGPUS" --block &
  
  # Wait for Ray cluster to initialize
  echo "⏳ Waiting for Ray cluster to initialize..."
  sleep 30
  
  # Check Ray cluster status
  if ! check_ray_ready; then
    echo "❌ Failed to start Ray cluster"
    exit 1
  fi
  
  echo "📊 Ray Dashboard: http://$POD_IP:8265"
  echo "🎯 == Submitting DAPO training job to Ray cluster =="
  
  # Submit training job using ray job submit
  export HYDRA_FULL_ERROR=1
  JOB_ID="dapo-training-$(date +%s)"
  echo "📝 Submitting DAPO job with ID: $JOB_ID"
  
  ray job submit --address="http://$POD_IP:8265" \
    --no-wait \
    --job-id="$JOB_ID" \
    -- \
    python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.filter_groups.enable="${filter_groups_enable}" \
    algorithm.filter_groups.metric="${filter_groups_metric}" \
    algorithm.filter_groups.max_num_gen_batches="${max_num_gen_batches}" \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    data.gen_batch_size="${gen_batch_size}" \
    data.train_batch_size="${train_batch_size}" \
    data.val_batch_size="${val_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts="${filter_overlong_prompts}" \
    data.truncation="${truncation}" \
    data.dataloader_num_workers="${dataloader_num_workers}" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing="${enable_gradient_checkpointing}" \
    actor_rollout_ref.model.use_remove_padding="${use_remove_padding}" \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c="${clip_ratio_c}" \
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
    actor_rollout_ref.rollout.val_kwargs.n="${val_rollout_n}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${val_do_sample}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable="${enable_overlong_buffer}" \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len="${overlong_buffer_len}" \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor="${overlong_penalty_factor}" \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len="${max_response_length}" \
    trainer.logger='["console", "swanlab"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.critic_warmup="${critic_warmup}" \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq="${save_freq}" \
    trainer.val_before_train="${val_before_train}" \
    trainer.n_gpus_per_node="${ngpus_per_node}" \
    trainer.nnodes="${nnodes}" \
    trainer.default_local_dir="${output_dir}" \
    "$@"
  
  # Wait for job to start and track logs
  echo "🔍 Waiting for job to start..."
  sleep 10
  
  # Verify job status
  echo "📋 Current jobs:"
  ray job list
  
  echo "📊 Job ID: $JOB_ID"
  echo "📋 Following logs for job: $JOB_ID"
  
  # Create status file directory
  STATUS_DIR="/tmp/ray_job_status"
  mkdir -p "$STATUS_DIR"
  STATUS_FILE="$STATUS_DIR/job_status"
  
  # Initialize status file
  echo "RUNNING" > "$STATUS_FILE"
  
  # Background job status monitoring
  monitor_job_status() {
    local job_id="$1"
    while true; do
      sleep 30
      local status=$(ray job status "$job_id" 2>/dev/null | grep -o "Status: [A-Z]*" | cut -d' ' -f2 || echo "UNKNOWN")
      echo "$status" > "$STATUS_FILE"
      
      if [[ "$status" == "SUCCEEDED" || "$status" == "FAILED" || "$status" == "STOPPED" ]]; then
        echo "🎯 Job finished with status: $status"
        break
      fi
    done
  }
  
  # Start background monitoring
  monitor_job_status "$JOB_ID" &
  MONITOR_PID=$!
  
  # Follow logs and handle job completion
  ray job logs "$JOB_ID" --follow || {
    echo "❌ Failed to follow logs for job: $JOB_ID"
    echo "🔍 Checking job status:"
    ray job status "$JOB_ID" || echo "Job status check failed"
    echo ""
    echo "🔧 Try manually with: ray job logs $JOB_ID --follow"
    echo "🔧 Or check all jobs: ray job list"
    echo "📊 Dashboard: http://$POD_IP:8265"
    
    # Set failure status
    echo "FAILED" > "$STATUS_FILE"
  }
  
  # Wait for monitoring process to finish
  wait $MONITOR_PID 2>/dev/null || true
  
  # Read final status
  FINAL_STATUS=$(cat "$STATUS_FILE" 2>/dev/null || echo "UNKNOWN")
  echo "🏁 DAPO training job completed with status: $FINAL_STATUS"
  
  # Cleanup Ray cluster
  echo "🧹 Cleaning up Ray cluster..."
  ray stop --force || true
  
  # Set exit code based on status
  if [[ "$FINAL_STATUS" == "SUCCEEDED" ]]; then
    echo "✅ DAPO Multi-node training completed successfully!"
    echo "📁 Checkpoints saved to: ${output_dir}"
    echo "📊 Experiment name: ${experiment_name}"
    exit 0
  else
    echo "❌ DAPO training failed with status: $FINAL_STATUS"
    exit 1
  fi

else
  echo "🔗 == Starting Ray WORKER Node -> $HEAD_SVC:6379 =="
  
  # Wait for master Ray service to start
  if ! wait_for_master; then
    echo "❌ Failed to connect to master Ray service"
    exit 1
  fi
  
  echo "🚀 Starting Ray worker..."
  ray start --address "$HEAD_SVC:6379" --node-ip-address "$POD_IP" --num-gpus "$NGPUS" --block &
  
  # Wait for connection
  echo "⏳ Waiting for worker to connect..."
  sleep 15
  
  # Verify connection status
  if ray status >/dev/null 2>&1; then
    echo "✅ Worker successfully connected to Ray cluster!"
    ray status
  else
    echo "⚠️  Worker connection status unclear, but continuing..."
  fi
  
  echo "🎯 Worker node ready, waiting for training job assignment..."
  echo "💡 This worker will automatically participate in DAPO distributed training"
  echo "📊 Monitor training progress at: http://$HEAD_SVC:8265"
  
  # Monitor master node status
  monitor_master_status() {
    local status_file="/tmp/ray_job_status/job_status"
    local check_interval=30
    local max_connection_failures=5
    local connection_failures=0
    
    echo "🔍 Starting master node status monitoring..."
    
    while true; do
      sleep $check_interval
      
      # Check Ray cluster connection status
      if ! ray status >/dev/null 2>&1; then
        ((connection_failures++))
        echo "⚠️  Ray cluster connection lost (failure $connection_failures/$max_connection_failures)"
        
        if [ $connection_failures -ge $max_connection_failures ]; then
          echo "❌ Ray cluster connection permanently lost, shutting down worker"
          break
        fi
        continue
      else
        # Reset connection failure count
        connection_failures=0
      fi
      
      # Try to check master node status file via network
      if timeout 10 bash -c "test -f $status_file" 2>/dev/null; then
        local status=$(cat "$status_file" 2>/dev/null || echo "UNKNOWN")
        echo "📊 Master job status: $status"
        
        if [[ "$status" == "SUCCEEDED" || "$status" == "FAILED" || "$status" == "STOPPED" ]]; then
          echo "🎯 Master job finished with status: $status, shutting down worker"
          break
        fi
      else
        # If can't access status file, check if master Ray service is still running
        if ! timeout 5 bash -c "</dev/tcp/$HEAD_SVC/6379" 2>/dev/null; then
          echo "🔌 Master Ray service is no longer accessible"
          ((connection_failures++))
          
          if [ $connection_failures -ge $max_connection_failures ]; then
            echo "❌ Master node appears to be down, shutting down worker"
            break
          fi
        else
          connection_failures=0
        fi
      fi
    done
    
    echo "🏁 Worker monitoring completed"
  }
  
  # Start background monitoring
  monitor_master_status &
  MONITOR_PID=$!
  
  # Set signal handling for graceful exit
  cleanup_worker() {
    echo "🧹 Cleaning up worker node..."
    kill $MONITOR_PID 2>/dev/null || true
    ray stop --force || true
    exit 0
  }
  
  trap cleanup_worker SIGTERM SIGINT
  
  # Wait for monitoring process to finish
  wait $MONITOR_PID
  
  # Cleanup Ray worker
  echo "🧹 Stopping Ray worker..."
  ray stop --force || true
fi
