# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value
    # Optional per-sample scaling for KL-in-reward
    scale = None
    try:
        if "kl_in_reward_scale" in data.non_tensor_batch:
            scale_np = data.non_tensor_batch["kl_in_reward_scale"]
            import numpy as _np
            if isinstance(scale_np, _np.ndarray):
                scale = torch.as_tensor(scale_np, device=token_level_scores.device, dtype=token_level_scores.dtype)
            else:
                scale = torch.as_tensor(scale_np, device=token_level_scores.device, dtype=token_level_scores.dtype)
    except Exception:
        scale = None

    if scale is not None:
        beta_mat = beta * scale.unsqueeze(-1)
        token_level_rewards = token_level_scores - beta_mat * kld
    else:
        token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    
    # Compute separate KL metrics for hard and non-hard samples
    metrics = {}
    
    # Check if we have difficulty classification
    if "difficulty_source" in data.non_tensor_batch:
        difficulty_source = data.non_tensor_batch["difficulty_source"]
        
        # Separate hard and non-hard samples
        hard_mask = np.array([str(s) == "hard" for s in difficulty_source])
        non_hard_mask = ~hard_mask
        
        if np.any(hard_mask):
            hard_kl = current_kl[torch.from_numpy(hard_mask).to(current_kl.device)]
            hard_kl_mean = torch.mean(hard_kl).item() if len(hard_kl) > 0 else 0.0
            metrics["actor/reward_kl_penalty_hard"] = hard_kl_mean
            if scale is not None:
                hard_scale = scale[torch.from_numpy(hard_mask).to(scale.device)]
                metrics["actor/reward_kl_coeff_hard"] = (beta * torch.mean(hard_scale)).item() if len(hard_scale) > 0 else beta
            else:
                metrics["actor/reward_kl_coeff_hard"] = beta
        
        if np.any(non_hard_mask):
            non_hard_kl = current_kl[torch.from_numpy(non_hard_mask).to(current_kl.device)]
            non_hard_kl_mean = torch.mean(non_hard_kl).item() if len(non_hard_kl) > 0 else 0.0
            metrics["actor/reward_kl_penalty_non_hard"] = non_hard_kl_mean
            if scale is not None:
                non_hard_scale = scale[torch.from_numpy(non_hard_mask).to(scale.device)]
                metrics["actor/reward_kl_coeff_non_hard"] = (beta * torch.mean(non_hard_scale)).item() if len(non_hard_scale) > 0 else beta
            else:
                metrics["actor/reward_kl_coeff_non_hard"] = beta
    
    # Overall metrics
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics["actor/reward_kl_penalty"] = current_kl
    metrics["actor/reward_kl_penalty_coeff"] = beta

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    return data, metrics


def classify_samples_by_difficulty(data: DataProto, hard_acc_upper: float = 0.3, med_acc_lower: float = 0.3, med_acc_upper: float = 0.7) -> DataProto:
    """Classify samples by difficulty based on reward/accuracy and add difficulty labels.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        hard_acc_upper (float): Upper threshold for hard samples. Defaults to 0.3.
        med_acc_lower (float): Lower threshold for medium samples. Defaults to 0.3.
        med_acc_upper (float): Upper threshold for medium samples. Defaults to 0.7.

    Returns:
        DataProto: The updated data with difficulty classification labels.
    """
    uid = data.non_tensor_batch["uid"]
    
    if "acc" in data.batch.keys():
        acc_arr = np.asarray([x.item() if torch.is_tensor(x) else float(x) for x in data.batch["acc"]], dtype=float)
    elif "acc" in data.non_tensor_batch.keys():
        acc_arr = np.asarray(data.non_tensor_batch["acc"], dtype=float)
    else:
        raise ValueError("No 'acc' field found in batch data. Accuracy is required for difficulty classification.")

    # Compute per-uid accuracy
    uid2acc = {}
    for i, u in enumerate(uid):
        uid2acc.setdefault(u, []).append(float(acc_arr[i]))
    uid2acc = {u: float(np.mean(v)) for u, v in uid2acc.items()}

    # Classify samples by difficulty
    difficulty_labels = []
    for i, u in enumerate(uid):
        acc = uid2acc[u]
        if acc <= hard_acc_upper:
            difficulty_labels.append("hard")
        elif med_acc_lower < acc <= med_acc_upper:
            difficulty_labels.append("medium")
        else:
            difficulty_labels.append("easy")
    
    # Add difficulty source to non_tensor_batch
    data.non_tensor_batch["difficulty_source"] = np.array(difficulty_labels, dtype=object)
    
    return data


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        
        # For GRPO with difficulty classification, we need to classify samples first
        # based on their reward mean (similar to CLPO)
        if config and config.get("enable_difficulty_classification", False):
            # Classify samples by reward mean for GRPO
            uid = data.non_tensor_batch["uid"]
            seq_rewards = data.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
            
            # Compute per-uid accuracy/reward mean
            uid2reward = {}
            for i, u in enumerate(uid):
                uid2reward.setdefault(u, []).append(float(seq_rewards[i]))
            uid2reward_mean = {u: float(np.mean(v)) for u, v in uid2reward.items()}
            
            # Get thresholds from config
            hard_reward_upper = config.get("hard_reward_upper", 0.3)  # Reward threshold for hard samples
            med_reward_lower = config.get("med_reward_lower", 0.3)
            med_reward_upper = config.get("med_reward_upper", 0.7)
            
            # Classify samples by difficulty
            difficulty_labels = []
            for i, u in enumerate(uid):
                reward_mean = uid2reward_mean[u]
                if reward_mean <= hard_reward_upper:
                    difficulty_labels.append("hard")
                elif med_reward_lower < reward_mean <= med_reward_upper:
                    difficulty_labels.append("medium")
                else:
                    difficulty_labels.append("easy")
            
            # Add difficulty source to non_tensor_batch
            data.non_tensor_batch["difficulty_source"] = np.array(difficulty_labels, dtype=object)
            
            # Add KL scaling based on difficulty classification
            try:
                difficulty_source = data.non_tensor_batch["difficulty_source"]
                
                # In-reward multipliers
                r_hard = float(config.get("kl_in_reward_coef_hard", 1.0))
                r_non = float(config.get("kl_in_reward_coef_nonhard", 1.0))
                r_scale = np.array([
                    r_hard if str(s) == "hard" else r_non for s in difficulty_source
                ], dtype=float)
                data.non_tensor_batch["kl_in_reward_scale"] = r_scale

                # In-loss multipliers
                l_hard = float(config.get("kl_loss_coef_hard_scale", 1.0))
                l_non = float(config.get("kl_loss_coef_nonhard_scale", 1.0))
                l_scale = torch.tensor([
                    l_hard if str(s) == "hard" else l_non for s in difficulty_source
                ], dtype=data.batch["token_level_rewards"].dtype, device=data.batch["token_level_rewards"].device)
                data.batch["kl_in_loss_scale"] = l_scale
            except Exception as _e:
                print(f"[DCKL GRPO] scale preparation failed: {_e}")
        
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        
        # Best model tracking for conditional saving
        self.best_metrics = {}
        self.save_best_model = getattr(config.trainer, 'save_best_model', False)
        self.best_metric_names = getattr(config.trainer, 'best_metric_names', [])
        if self.save_best_model and not self.best_metric_names:
            # Default to common core metrics
            self.best_metric_names = ['val-core/mean@1']

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        
        # Compute pass@k metrics
        pass_k_metrics = self._compute_pass_k_metrics(sample_uids, reward_extra_infos_dict)
        
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best", "pass"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        
        # Add pass@k metrics to core metrics
        for metric_name, metric_val in pass_k_metrics.items():
            metric_dict[f"val-core/{metric_name}"] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _is_best_model(self, val_metrics: dict[str, float]) -> bool:
        """Check if current validation metrics represent the best model so far.
        
        Args:
            val_metrics: Dictionary of validation metrics
            
        Returns:
            True if any tracked metric achieved a new best value
        """
        if not self.save_best_model or not self.best_metric_names:
            return False
            
        is_best = False
        for metric_name in self.best_metric_names:
            if metric_name in val_metrics:
                current_value = val_metrics[metric_name]
                if metric_name not in self.best_metrics or current_value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = current_value
                    is_best = True
                    print(f"New best {metric_name}: {current_value:.4f}")
        
        return is_best

    def _compute_pass_k_metrics(self, sample_uids: list[str], reward_extra_infos_dict: dict[str, list]) -> dict[str, float]:
        """Compute pass@k metrics for validation.
        
        pass@k means that at least 1 out of k attempts is correct for each unique problem.
        
        Args:
            sample_uids: List of sample UIDs (may have duplicates for multiple attempts per problem)
            reward_extra_infos_dict: Dictionary containing reward information
            
        Returns:
            Dictionary of pass@k metrics
        """
        metrics = {}
        
        # Get accuracy information
        if "acc" in reward_extra_infos_dict:
            acc_data = reward_extra_infos_dict["acc"]
        else:
            # Fallback: use reward > 0 as accuracy
            rewards = reward_extra_infos_dict.get("reward", [])
            acc_data = [1.0 if r > 0 else 0.0 for r in rewards]
        
        if not acc_data or len(acc_data) != len(sample_uids):
            return metrics
            
        # Group by unique problem (assuming UIDs without rollout suffix are unique problems)
        # For validation, we typically generate multiple responses per problem
        uid_to_accs = {}
        for uid, acc in zip(sample_uids, acc_data):
            # Extract base UID (remove rollout numbering if present)
            base_uid = str(uid).split('_rollout_')[0] if '_rollout_' in str(uid) else str(uid)
            if base_uid not in uid_to_accs:
                uid_to_accs[base_uid] = []
            uid_to_accs[base_uid].append(float(acc))
        
        if not uid_to_accs:
            return metrics
            
        # Calculate pass@k for different k values
        k_values = [1, 2, 4, 8, 16, 32]        
        for k in k_values:
            pass_at_k_scores = []
            for base_uid, accs in uid_to_accs.items():
                if len(accs) >= k:
                    # pass@k: at least 1 correct in first k attempts
                    k_accs = accs[:k]
                    pass_at_k = 1.0 if any(acc > 0.5 for acc in k_accs) else 0.0
                    pass_at_k_scores.append(pass_at_k)
            
            if pass_at_k_scores:
                metrics[f"pass@{k}"] = np.mean(pass_at_k_scores)
        
        # Also compute pass@max (use all available attempts)
        pass_at_max_scores = []
        for base_uid, accs in uid_to_accs.items():
            pass_at_max = 1.0 if any(acc > 0.5 for acc in accs) else 0.0
            pass_at_max_scores.append(pass_at_max)
        
        if pass_at_max_scores:
            max_k = max(len(accs) for accs in uid_to_accs.values())
            metrics[f"pass@{max_k}"] = np.mean(pass_at_max_scores)
        
        return metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Classify samples by difficulty based on reward/accuracy
                        if self.config.algorithm.get("enable_difficulty_classification", False):
                            hard_acc_upper = self.config.algorithm.get("hard_acc_upper", 0.3)
                            med_acc_lower = self.config.algorithm.get("med_acc_lower", 0.3) 
                            med_acc_upper = self.config.algorithm.get("med_acc_upper", 0.7)
                            batch = classify_samples_by_difficulty(
                                batch, hard_acc_upper=hard_acc_upper, 
                                med_acc_lower=med_acc_lower, med_acc_upper=med_acc_upper
                            )
                            
                            # Add KL scaling based on difficulty classification
                            try:
                                difficulty_source = batch.non_tensor_batch.get("difficulty_source", np.array(["medium"] * len(batch), dtype=object))
                                
                                # In-reward multipliers
                                r_hard = float(self.config.algorithm.get("kl_in_reward_coef_hard", 1.0))
                                r_non = float(self.config.algorithm.get("kl_in_reward_coef_nonhard", 1.0))
                                r_scale = np.array([
                                    r_hard if str(s) == "hard" else r_non for s in difficulty_source
                                ], dtype=float)
                                batch.non_tensor_batch["kl_in_reward_scale"] = r_scale

                                # In-loss multipliers
                                l_hard = float(self.config.actor_rollout_ref.actor.get("kl_loss_coef_hard_scale", 1.0))
                                l_non = float(self.config.actor_rollout_ref.actor.get("kl_loss_coef_nonhard_scale", 1.0))
                                l_scale = torch.tensor([
                                    l_hard if str(s) == "hard" else l_non for s in difficulty_source
                                ], dtype=batch.batch["token_level_scores"].dtype, device=batch.batch["token_level_scores"].device)
                                batch.batch["kl_in_loss_scale"] = l_scale
                            except Exception as _e:
                                print(f"[DCKL] scale preparation failed: {_e}")

                        # Add difficulty distribution statistics for PPO
                        self._add_difficulty_stats(batch, metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    
                    # Check if this is the best model and save if needed
                    if self._is_best_model(val_metrics):
                        with marked_timer("save_best_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                            print("Saved best model checkpoint")

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def _add_difficulty_stats(self, batch: DataProto, metrics: dict):
        """Add difficulty distribution statistics for PPO training monitoring."""
        try:
            # Get accuracy information
            if "acc" in batch.batch.keys():
                acc_arr = np.asarray([x.item() if torch.is_tensor(x) else float(x) for x in batch.batch["acc"]], dtype=float)
            elif "acc" in batch.non_tensor_batch.keys():
                acc_arr = np.asarray(batch.non_tensor_batch["acc"], dtype=float)
            else:
                raise ValueError("No 'acc' field found in batch data. Accuracy is required for difficulty classification.")

            # Get UIDs
            uid = batch.non_tensor_batch["uid"]
            
            # Compute per-uid accuracy
            uid2acc = {}
            for i, u in enumerate(uid):
                uid2acc.setdefault(u, []).append(float(acc_arr[i]))
            uid2acc = {u: float(np.mean(v)) for u, v in uid2acc.items()}

            # Define difficulty thresholds (using CLPO defaults if available, otherwise reasonable defaults)
            hard_threshold = getattr(self, 'clpo_hard_acc_upper', 0.3)
            med_lower = getattr(self, 'clpo_med_acc_lower', 0.3)
            med_upper = getattr(self, 'clpo_med_acc_upper', 0.7)

            # Classify by difficulty
            hard_uids = [u for u, a in uid2acc.items() if a <= hard_threshold]
            med_uids = [u for u, a in uid2acc.items() if med_lower < a <= med_upper]
            easy_uids = [u for u, a in uid2acc.items() if a > med_upper]

            # Calculate ratios
            total_samples = len(batch)
            metrics["ppo_batch/easy_ratio%"] = round(100 * len(easy_uids) / max(1, total_samples), 2)
            metrics["ppo_batch/medium_ratio%"] = round(100 * len(med_uids) / max(1, total_samples), 2)
            metrics["ppo_batch/hard_ratio%"] = round(100 * len(hard_uids) / max(1, total_samples), 2)

            # Calculate accuracy statistics for each difficulty level
            def calc_difficulty_stats(uids: list[str], difficulty_name: str) -> dict:
                if not uids:
                    return {
                        f"ppo_batch/{difficulty_name}_count": 0,
                        f"ppo_batch/{difficulty_name}_acc_mean": 0.0,
                        f"ppo_batch/{difficulty_name}_acc_std": 0.0
                    }
                
                accs = [uid2acc[uid] for uid in uids]
                return {
                    f"ppo_batch/{difficulty_name}_count": len(uids),
                    f"ppo_batch/{difficulty_name}_acc_mean": round(float(np.mean(accs)), 4),
                    f"ppo_batch/{difficulty_name}_acc_std": round(float(np.std(accs)), 4)
                }

            # Add difficulty-specific statistics
            metrics.update(calc_difficulty_stats(easy_uids, "easy"))
            metrics.update(calc_difficulty_stats(med_uids, "medium"))
            metrics.update(calc_difficulty_stats(hard_uids, "hard"))

            # Overall accuracy statistics
            all_accs = list(uid2acc.values())
            metrics["ppo_batch/overall_acc_mean"] = round(float(np.mean(all_accs)), 4)
            metrics["ppo_batch/overall_acc_std"] = round(float(np.std(all_accs)), 4)

        except Exception as e:
            print(f"[WARNING] Failed to compute difficulty stats: {e}")
            # Add default values to avoid missing metrics
            metrics.update({
                "ppo_batch/easy_ratio%": 0.0,
                "ppo_batch/medium_ratio%": 0.0,
                "ppo_batch/hard_ratio%": 0.0,
                "ppo_batch/overall_acc_mean": 0.0,
                "ppo_batch/overall_acc_std": 0.0
            })

class RayCLPOTrainer(RayPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # CLPO thresholds from config
        data_cfg = getattr(self.config, "data", {})
        self.clpo_hard_acc_upper = float(getattr(data_cfg, "clpo_hard_acc_upper", 0.3))
        self.clpo_med_acc_lower = float(getattr(data_cfg, "clpo_medium_acc_lower", 0.3))
        self.clpo_med_acc_upper = float(getattr(data_cfg, "clpo_medium_acc_upper", 0.7))
        self.max_prompt_length = int(getattr(data_cfg, "max_prompt_length", 2048))
        
        # CLPO rewrite data saving configuration
        self.clpo_save_rewrite_data = getattr(data_cfg, "clpo_save_rewrite_data", False)
        self.clpo_rewrite_save_path = getattr(data_cfg, "clpo_rewrite_save_path", None)
        self.clpo_hard_rewrite_save_path = getattr(data_cfg, "clpo_hard_rewrite_save_path", None)
        self.clpo_medium_rewrite_save_path = getattr(data_cfg, "clpo_medium_rewrite_save_path", None)
        self.rewrite_data_buffer = []  # Buffer to store all rewrite data
        self.hard_rewrite_data_buffer = []  # Buffer to store hard rewrite data
        self.medium_rewrite_data_buffer = []  # Buffer to store medium rewrite data
        
        # CLPO ablation study configuration
        self.clpo_rewrite_mode = getattr(data_cfg, "clpo_rewrite_mode", "both")  # "hard_only", "medium_only", "both"
        print(f"[CLPO] Rewrite mode: {self.clpo_rewrite_mode}")

        # Rewrite instruction templates
        self.rewrite_instr_hard = (
            "You are a master-level question rewriting expert. Your mission is TOP-SECRET: rewrite the given math question so that it is simpler, clearer, and more direct, while maintaining the EXACT same task, constraints, solution method, and CORRECT ANSWER. DO NOT solve the problem or make any assumptions about its solution.\n\n"
            "CRITICAL RULES FOR OUTPUT:\n"
            "- OUTPUT MUST BE A SINGLE CODE BLOCK using the exact format provided below:\n"
            "```text\n"
            "user\n"
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            "[Rewritten question here]\n\n"
            "Remember to put your answer on its own line after \"Answer:\".\n"
            "assistant\n"
            "```\n"
            "- Output NOTHING outside the code block.\n"
            "- Do NOT include any reasoning, parsing, or explanation of the problem (e.g., avoid phrases like \"Let me simplify this question\" or \"This problem can be rewritten as...\").\n"
            "- Rewrite the question to make it SIMPLER and CLEARER for the reader, while preserving the EXACT task, math content, and structure.\n"
            "- Keep the rewritten question STRICTLY UNDER 400 tokens, including all characters, symbols, and spaces.\n"
            "- Remove redundancy and completely clarify definitions, if needed.** For example, if there are implied constraints, make them explicit.\n"
            "- Preserve ALL formatting or directive rules exactly as in the original question (e.g., 'write the answer in a box', 'solve step by step').\n"
            "- The rewritten question must lead to the SAME correct answer as the original: {ANSWER}. DO NOT explicitly, implicitly, or indirectly REVEAL or SUGGEST the answer.\n\n"
            "HOW TO SIMPLIFY:\n"
            "1) Eliminate redundant or overly verbose phrasing while keeping the exact math concepts, symbols, and structure.\n"
            "2) Avoid complex clauses by breaking them into shorter, simpler sentences, but do NOT remove or alter necessary constraints.\n"
            "3) Include IMPLICIT constraints explicitly if they improve clarity (e.g., 'x > 0' if it is implied implicitly).\n"
            "4) DO NOT add any background information, new context, or reasoning unrelated to the question.\n\n"
            "ONE-SHOT EXAMPLE (STRICT FORMAT ONLY):\n\n"
            "Original question (with role markers):\n"
            "user\n"
            "Compute the value of 2 + 3. The last line of your response should be of the form Answer: $Answer.\n\n"
            "Remember to put your answer on its own line after \"Answer:\".\n"
            "assistant\n\n"
            "Your output:\n"
            "```text\n"
            "user\n"
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            "What is the sum of 2 and 3?\n\n"
            "Remember to put your answer on its own line after \"Answer:\".\n"
            "assistant\n"
            "```\n\n"
            "ORIGINAL_QUESTION (to rewrite):\n"
            "{Q}\n"
        )
        self.rewrite_instr_med = (
            "You are a master-level question rewriting expert. Your mission is TOP-SECRET: rewrite the given math question into a diverse but semantically equivalent version, while maintaining the EXACT same task, constraints, solution method, and CORRECT ANSWER. DO NOT solve the problem or make any assumptions about its solution.\n\n"
            "CRITICAL RULES FOR OUTPUT:\n"
            "- OUTPUT MUST BE A SINGLE CODE BLOCK using the exact format provided below:\n"
            "```text\n"
            "user\n"
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            "[Rewritten question here]\n\n"
            "Remember to put your answer on its own line after \"Answer:\".\n"
            "assistant\n"
            "```\n"
            "- Output NOTHING outside the code block.\n"
            "- Rewrite the question to DIVERSIFY its expression but PRESERVE its exact meaning and constraints.\n"
            "- Keep the rewritten question STRICTLY UNDER 400 tokens, including all characters, symbols, and spaces.\n"
            "- DO NOT change any math content (variables, relationships, solution method, etc.).\n"
            "- Avoid repetitive sentence structures or template-like phrasing by using varied grammar structure, synonyms, or logical order.\n"
            "- Preserve ALL formatting or directive rules exactly as in the original question (e.g., 'write the answer in a box', 'solve step by step').\n"
            "- The rewritten question must lead to the SAME correct answer as the original: {ANSWER}. DO NOT explicitly, implicitly, or indirectly REVEAL or SUGGEST the answer.\n\n"
            "HOW TO DIVERSIFY:\n"
            "1) Rephrase clauses logically (e.g., change phrasing while maintaining meaning).\n"
            "   * Original: 'What is the sum of 4 and 5?' ➡ 'Calculate the result of adding 4 with 5.'\n"
            "2) Rearrange sentence structure without altering mathematical intent.\n"
            "   * Original: 'Evaluate 3x - 1 where x = 2.' ➡ 'Find the value of 3x minus 1, with x set to 2.'\n"
            "3) Use synonyms or alternative phrasing to express the same logic and tasks.\n"
            "   * Original: 'Determine the area of the rectangle.' ➡ 'Find the rectangular region's area.'\n"
            "4) DO NOT add or remove constraints, context, or instructions.\n\n"
            "ONE-SHOT EXAMPLE (STRICT FORMAT ONLY):\n\n"
            "Original question (with role markers):\n"
            "user\n"
            "Compute the value of 2 + 3. The last line of your response should be of the form Answer: $Answer.\n\n"
            "Remember to put your answer on its own line after \"Answer:\".\n"
            "assistant\n\n"
            "Your output:\n"
            "```text\n"
            "user\n"
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            "Determine the result of adding 2 and 3.\n\n"
            "Remember to put your answer on its own line after \"Answer:\".\n"
            "assistant\n"
            "```\n\n"
            "ORIGINAL_QUESTION (to rewrite):\n"
            "{Q}\n"
        )

    def _extract_ground_truth(self, batch: DataProto, idxs: list[int]) -> list[str]:
        """Extract correct answers from batch for specified indices"""
        
        if not idxs:
            return []
        
        answers = []
        
        if hasattr(batch, '_reward_extra_infos') and batch._reward_extra_infos:
            if 'ground_truth' in batch._reward_extra_infos:
                gt_data = batch._reward_extra_infos['ground_truth']
                return [str(gt_data[i]) for i in idxs]
            elif 'expected_answer' in batch._reward_extra_infos:
                gt_data = batch._reward_extra_infos['expected_answer']
                return [str(gt_data[i]) for i in idxs]
        
        if "reward_model" in batch.non_tensor_batch:
            reward_model_data = batch.non_tensor_batch["reward_model"]
            
            if isinstance(reward_model_data, dict):
                for key in ['ground_truth', 'answer', 'expected', 'target', 'correct_answer']:
                    if key in reward_model_data:
                        gt_data = reward_model_data[key]
                        if isinstance(gt_data, (list, np.ndarray)) and len(gt_data) > max(idxs):
                            result = [str(gt_data[i]) for i in idxs]
                            return result
            
            elif isinstance(reward_model_data, (list, np.ndarray)) and len(reward_model_data) > max(idxs):
                raw_result = [str(reward_model_data[i]) for i in idxs]
                
                parsed_result = []
                for item in raw_result:
                    try:
                        # Handle string representation of dict
                        if item.startswith("{'") and item.endswith("'}"):
                            # Convert single quotes to double quotes for JSON parsing
                            json_str = item.replace("'", '"')
                            import json
                            parsed = json.loads(json_str)
                            if 'ground_truth' in parsed:
                                parsed_result.append(str(parsed['ground_truth']))
                            else:
                                parsed_result.append(item)  # Fallback to original
                        else:
                            parsed_result.append(item)  # Not a JSON-like string
                    except Exception as e:
                        parsed_result.append(item)  # Fallback to original
                
                return parsed_result
        
        for field_name in ['ground_truth', 'answers', 'targets', 'expected_answers']:
            if field_name in batch.non_tensor_batch:
                gt_data = batch.non_tensor_batch[field_name]
                if isinstance(gt_data, (list, np.ndarray)) and len(gt_data) > max(idxs):
                    result = [str(gt_data[i]) for i in idxs]
                    return result
    
        print(f"[WARNING] Could not find ground truth for indices {idxs}")
        return ["[ANSWER_NOT_FOUND]"] * len(idxs)

    def _decode_prompt_text(self, batch: DataProto, idx: int) -> str:
        """Extract prompt text from DataProto"""
        try:
            if "raw_prompt_ids" in batch.non_tensor_batch and idx < len(batch.non_tensor_batch["raw_prompt_ids"]):
                ids = batch.non_tensor_batch["raw_prompt_ids"][idx]
                if isinstance(ids, np.ndarray):
                    ids = ids.tolist()
                return self.tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            pass

        try:
            if batch.batch is not None and "prompts" in batch.batch:
                return self.tokenizer.decode(batch.batch["prompts"][idx], skip_special_tokens=True)
        except Exception:
            pass

        if batch.batch is not None and "input_ids" in batch.batch:
            full_ids = batch.batch["input_ids"][idx]
            if "responses" in batch.batch:
                resp_len = int(batch.batch["responses"][idx].shape[-1])
                prompt_ids = full_ids[..., :-resp_len] if resp_len > 0 else full_ids
            else:
                prompt_ids = full_ids
            return self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        
        return ""

    def _build_prompt_batch(self, rewritten_texts: list[str]) -> DataProto:
        """Build DataProto batch from rewritten prompt texts"""
        if not rewritten_texts:
            return None
            
        model_inputs = self.tokenizer(
            rewritten_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.get("truncation", "error"),
        )
        
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = np.array(
            self.tokenizer.batch_encode_plus(rewritten_texts, add_special_tokens=False)["input_ids"],
            dtype=object,
        )
        tools_kwargs = np.array([{} for _ in range(len(position_ids))], dtype=object)
        
        gen_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "tools_kwargs": tools_kwargs,
        }
        return DataProto.from_single_dict(gen_dict)

    def _generate_rewritten_questions(self, rewrite_prompts: list[str], original_batch: DataProto, original_idxs: list[int]) -> tuple[list[str], DataProto]:
        """Generate ONE rewritten question per original question using the rewrite prompts"""
        if not rewrite_prompts:
            return [], None
            
        prompt_batch = self._build_prompt_batch(rewrite_prompts)
        if prompt_batch is None:
            return [], None
            
        gen_batch = self._get_gen_batch(prompt_batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        
        gen_batch.meta_info["rewrite_mode"] = True
        gen_batch.meta_info["max_new_tokens"] = 400  
        gen_batch.meta_info["temperature"] = 0.7  
        gen_batch.meta_info["do_sample"] = True
        
        dp_size = (
            self.actor_rollout_wg.world_size
            if not getattr(self, "async_rollout_mode", False)
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, dp_size)
        
        if not getattr(self, "async_rollout_mode", False):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_padded)
        
        gen_batch_output = unpad_dataproto(gen_batch_output, pad_size=pad_size)
        
        raw_rewritten_responses = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_batch_output.batch["responses"]]
        
        rewritten_questions = []
        valid_indices = []
        
        for idx, response in enumerate(raw_rewritten_responses):
            try:
                # Primary method: extract from ```text``` blocks
                if "```text" in response:
                    extracted_question = response.split("```text")[1].split("```")[0].strip()
                # Fallback 1: extract from generic ``` blocks
                elif "```" in response:
                    extracted_question = response.split("```")[1].split("```")[0].strip()
                # Fallback 2: extract everything after the last instruction (if no code blocks)
                elif "Provide your rewritten question formatted strictly as:" in response:
                    extracted_question = response.split("Provide your rewritten question formatted strictly as:")[-1].strip()
                # Fallback 3: take the entire response if no clear format
                else:
                    extracted_question = response.strip()
                
                # Basic validation: check if the extracted question is reasonable
                if len(extracted_question) > 10 and len(extracted_question) < 2000:  # reasonable length
                    rewritten_questions.append(extracted_question)
                    valid_indices.append(idx)
                else:
                    print(f"[WARNING] Extracted question too short/long ({len(extracted_question)} chars), skipping")
                    
            except Exception as e:
                print(f"[ERROR] Failed to extract rewritten question from response {idx}: {e}")
                continue
        
        print(f"[DEBUG] Successfully extracted {len(rewritten_questions)}/{len(raw_rewritten_responses)} rewritten questions")
        
        # If no valid questions extracted, return empty
        if not rewritten_questions:
            print("[ERROR] No valid rewritten questions extracted")
            return [], None
        
        rewritten_batch_dict = {}
        
        # Use rewritten questions as new prompts (convert to numpy array)
        rewritten_batch_dict["prompts"] = np.array(rewritten_questions, dtype=object)
        
        model_inputs = self.tokenizer(
            rewritten_questions,  
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.get("truncation", "error"),
        )
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        rewritten_batch_dict["input_ids"] = input_ids
        rewritten_batch_dict["attention_mask"] = attention_mask
        rewritten_batch_dict["position_ids"] = position_ids
        
        mapped_original_idxs = [original_idxs[i] for i in valid_indices]
        
        for key, value in original_batch.non_tensor_batch.items():
            if key == "uid":
                continue
                
            if isinstance(value, dict):
                rewritten_dict = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (list, np.ndarray)) and len(subvalue) > 0:
                        selected_values = [subvalue[i] for i in mapped_original_idxs]
                        rewritten_dict[subkey] = np.array(selected_values, dtype=object)
                    else:
                        rewritten_dict[subkey] = subvalue
                rewritten_batch_dict[key] = rewritten_dict
            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                selected_values = [value[i] for i in mapped_original_idxs]
                rewritten_batch_dict[key] = np.array(selected_values, dtype=object)
            else:
                rewritten_batch_dict[key] = value
            
        rewritten_batch = DataProto.from_single_dict(rewritten_batch_dict)
        
        return rewritten_questions, rewritten_batch

    def _process_rewritten_batch_like_oqa(self, rewritten_batch: DataProto) -> DataProto:
        
        if rewritten_batch is None or len(rewritten_batch) == 0:
            return None
            
        rewritten_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(rewritten_batch))], dtype=object
        )
        
        gen_batch = self._get_gen_batch(rewritten_batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        
        dp_size = (
            self.actor_rollout_wg.world_size
            if not getattr(self, "async_rollout_mode", False)
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, dp_size)
        
        
        if not getattr(self, "async_rollout_mode", False):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_padded)
        
        gen_batch_output = unpad_dataproto(gen_batch_output, pad_size=pad_size)
        
        batch = rewritten_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        
        reward_tensor, reward_extra_infos = compute_reward(batch, self.reward_fn)
        batch.batch["reward_tensor"] = reward_tensor[1] if isinstance(reward_tensor, tuple) else reward_tensor
        if reward_extra_infos:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos.items()})
        
        return batch

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}


                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)


                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("original_gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            
                            del gen_baseline_batch, gen_baseline_output

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("oqa_reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        batch.batch["reward_tensor"] = reward_tensor
                        if reward_extra_infos_dict:
                            print(f"[DEBUG] reward_extra_infos_dict keys: {reward_extra_infos_dict.keys()}")
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    uid = batch.non_tensor_batch["uid"]
                    if "acc" in batch.batch.keys():
                        acc_arr = np.asarray([x.item() if torch.is_tensor(x) else float(x) for x in batch.batch["acc"]], dtype=float)
                    elif "acc" in batch.non_tensor_batch.keys():
                        acc_arr = np.asarray(batch.non_tensor_batch["acc"], dtype=float)
                    else:
                        seq_rewards = batch.batch["reward_tensor"].sum(dim=-1).cpu().numpy()
                        acc_arr = (seq_rewards > 0).astype(float)

                    uid2acc = {}
                    for i, u in enumerate(uid):
                        uid2acc.setdefault(u, []).append(float(acc_arr[i]))
                    uid2acc = {u: float(np.mean(v)) for u, v in uid2acc.items()}

                    hard_uids = [u for u, a in uid2acc.items() if a <= self.clpo_hard_acc_upper]
                    med_uids = [u for u, a in uid2acc.items() if self.clpo_med_acc_lower < a <= self.clpo_med_acc_upper]
                    oqa_train_uids = [u for u, a in uid2acc.items() if 0.0 < a < 1.0]

                    def uids_to_indices(uids: list[str]) -> list[int]:
                        return [i for i, u in enumerate(uid) if u in uids]

                    hard_idxs = uids_to_indices(hard_uids)
                    med_idxs = uids_to_indices(med_uids)
                    oqa_train_idxs = uids_to_indices(oqa_train_uids)



                    uid = batch.non_tensor_batch["uid"]
                    unique_uids = list(set(uid))
                    total_problem_count = len(unique_uids)
                    
                    uid2acc = {}
                    for i, u in enumerate(uid):
                        uid2acc.setdefault(u, []).append(float(acc_arr[i]))
                    uid2acc = {u: float(np.mean(v)) for u, v in uid2acc.items()}

                    easy_problem_count = 0
                    medium_problem_count = 0
                    hard_problem_count = 0
                    fully_correct_problem_count = 0
                    fully_wrong_problem_count = 0
                    
                    for u in unique_uids:
                        acc = uid2acc[u]
                        if acc == 1.0:
                            fully_correct_problem_count += 1
                        elif acc == 0.0:
                            fully_wrong_problem_count += 1
                        elif 0.0 < acc <= self.clpo_hard_acc_upper:
                            hard_problem_count += 1
                        elif self.clpo_med_acc_lower < acc <= self.clpo_med_acc_upper:
                            medium_problem_count += 1
                        elif self.clpo_med_acc_upper < acc < 1.0:
                            easy_problem_count += 1
                    
                    metrics["clpo/oqa_batch/total_problem_count"] = total_problem_count
                    metrics["clpo/oqa_batch/difficulty/easy_problem_count"] = easy_problem_count
                    metrics["clpo/oqa_batch/difficulty/medium_problem_count"] = medium_problem_count
                    metrics["clpo/oqa_batch/difficulty/hard_problem_count"] = hard_problem_count
                    

                    def collect_prompts(dp: DataProto, idxs: list[int]) -> list[str]:
                        return [self._decode_prompt_text(dp, i) for i in idxs]

                    hard_problem_uids = [u for u, a in uid2acc.items() if a <= self.clpo_hard_acc_upper]
                    medium_problem_uids = [u for u, a in uid2acc.items() if self.clpo_med_acc_lower < a <= self.clpo_med_acc_upper]
                    
                    should_rewrite_hard = self.clpo_rewrite_mode in ["hard_only", "both"]
                    should_rewrite_medium = self.clpo_rewrite_mode in ["medium_only", "both"]
                    
                    metrics["clpo/rewrite/selected_hard_for_rewrite_problem_count"] = len(hard_problem_uids) if should_rewrite_hard else 0
                    metrics["clpo/rewrite/selected_medium_for_rewrite_problem_count"] = len(medium_problem_uids) if should_rewrite_medium else 0
                    metrics["clpo/rewrite/rewrite_mode"] = self.clpo_rewrite_mode
                    
                    metrics["clpo/ablation/should_rewrite_hard"] = int(should_rewrite_hard)
                    metrics["clpo/ablation/should_rewrite_medium"] = int(should_rewrite_medium)
                    metrics["clpo/ablation/rewrite_mode_encoded"] = {
                        "hard_only": 1,
                        "medium_only": 2, 
                        "both": 3
                    }.get(self.clpo_rewrite_mode, 0)
                    
                    n_rollout = self.config.actor_rollout_ref.rollout.n
                    hard_unique_idxs = [idx for idx in hard_idxs if idx % n_rollout == 0] if should_rewrite_hard else []
                    med_unique_idxs = [idx for idx in med_idxs if idx % n_rollout == 0] if should_rewrite_medium else []

                    hard_prompts = collect_prompts(batch, hard_unique_idxs) if should_rewrite_hard else []
                    med_prompts = collect_prompts(batch, med_unique_idxs) if should_rewrite_medium else []

                    hard_answers = self._extract_ground_truth(batch, hard_unique_idxs) if hard_unique_idxs and should_rewrite_hard else []
                    med_answers = self._extract_ground_truth(batch, med_unique_idxs) if med_unique_idxs and should_rewrite_medium else []

                    def apply_rewrite(prompts: list[str], answers: list[str], mode: str) -> list[str]:
                        if not prompts:
                            return []
                        template = self.rewrite_instr_hard if mode == "hard" else self.rewrite_instr_med
                        
                        if len(answers) != len(prompts):
                            print(f"[WARNING] Mismatch: {len(prompts)} prompts vs {len(answers)} answers")
                            answers = answers[:len(prompts)] if len(answers) > len(prompts) else answers + ["[UNKNOWN]"] * (len(prompts) - len(answers))
                        
                        return [template.format(Q=p, ANSWER=a) for p, a in zip(prompts, answers)]

                    rewritten_hard_inputs = apply_rewrite(hard_prompts, hard_answers, mode="hard") if should_rewrite_hard else []
                    rewritten_med_inputs = apply_rewrite(med_prompts, med_answers, mode="med") if should_rewrite_medium else []
      
                    hard_out = None
                    med_out = None
                    hard_questions = []
                    med_questions = []
                    
                    if rewritten_hard_inputs:
                        hard_questions, hard_batch = self._generate_rewritten_questions(
                            rewritten_hard_inputs, batch, hard_unique_idxs
                        )
                        if hard_batch is not None:
                            hard_out = self._process_rewritten_batch_like_oqa(hard_batch)
                        else:
                            print(f"[DEBUG] ❌ none hard_batch")
                    else:
                        print(f"[DEBUG] ❌ none rewritten_hard_inputs")
                    
                    if rewritten_med_inputs:
                        med_questions, med_batch = self._generate_rewritten_questions(
                            rewritten_med_inputs, batch, med_unique_idxs
                        )
                        if med_batch is not None:
                            med_out = self._process_rewritten_batch_like_oqa(med_batch)
                        else:
                            print(f"[DEBUG] ❌ none med_batch")
                    else:
                        print(f"[DEBUG] ❌ none rewritten_med_inputs")
                    
                    
                    # Print Hard samples comparison
                    if hard_out is not None and len(hard_out) > 0:
                        hard_uids = hard_out.non_tensor_batch["uid"]
                        hard_responses = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in hard_out.batch["responses"]]
                        
                        hard_original_prompts = [hard_prompts[i] for i in range(min(1, len(hard_prompts)))]
                        hard_original_responses = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"][hard_unique_idxs[:min(1, len(hard_unique_idxs))]]]
                        
                        hard_rewritten_questions = [hard_questions[i] for i in range(min(1, len(hard_questions)))] if hard_questions else []
                        
                        for i in range(min(1, len(hard_uids))):
                            print(f"  Sample {i+1} (UID: {hard_uids[i]}):")
                            print(f"    ORIGINAL QUESTION: {hard_original_prompts[i]}")
                            print(f"    ORIGINAL RESPONSE: {hard_original_responses[i]}")
                            print(f"    REWRITTEN QUESTION: {hard_rewritten_questions[i]}")
                            print(f"    REWRITTEN RESPONSE: {hard_responses[i]}")
                            print()
                        
                        if self.clpo_save_rewrite_data:
                            self._add_rewrite_data_to_buffer(
                                hard_original_prompts, hard_original_responses,
                                hard_rewritten_questions, hard_responses, "hard"
                            )
                            self._save_rewrite_data_step()
                    
                    if med_out is not None and len(med_out) > 0:
                        print(f"[DEBUG] 🤖 MEDIUM REWRITE COMPARISON (showing first 1 samples):")
                        med_uids = med_out.non_tensor_batch["uid"]
                        med_responses = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in med_out.batch["responses"]]
                        
                        med_original_prompts = [med_prompts[i] for i in range(min(1, len(med_prompts)))]
                        med_original_responses = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"][med_unique_idxs[:min(1, len(med_unique_idxs))]]]
                        
                        med_rewritten_questions = [med_questions[i] for i in range(min(1, len(med_questions)))] if med_questions else []
                        
                        for i in range(min(1, len(med_uids))):
                            print(f"  Sample {i+1} (UID: {med_uids[i]}):")
                            print(f"    ORIGINAL QUESTION: {med_original_prompts[i]}")
                            print(f"    ORIGINAL RESPONSE: {med_original_responses[i]}")
                            print(f"    REWRITTEN QUESTION: {med_rewritten_questions[i]}")
                            print(f"    REWRITTEN RESPONSE: {med_responses[i]}")
                            print()
                        
                        if self.clpo_save_rewrite_data:
                            self._add_rewrite_data_to_buffer(
                                med_original_prompts, med_original_responses,
                                med_rewritten_questions, med_responses, "medium"
                            )
                            self._save_rewrite_data_step()

                    def keep_mid_acc(dp: DataProto) -> DataProto:
                        if dp is None or len(dp) == 0:
                            return None
                        seq_rewards = dp.batch["reward_tensor"].sum(dim=-1).cpu().numpy()
                        acc_arr = (seq_rewards > 0).astype(float)
                        uids = dp.non_tensor_batch["uid"]
                        uid2a = {}
                        for i, u in enumerate(uids):
                            uid2a.setdefault(u, []).append(float(acc_arr[i]))
                        keep_uids = [u for u, vals in uid2a.items() if 0.0 < float(np.mean(vals)) < 1.0]
                        keep_idxs = [i for i, u in enumerate(uids) if u in keep_uids]
                        return dp.select_idxs(keep_idxs) if keep_idxs else None

                    hard_keep = keep_mid_acc(hard_out)
                    med_keep = keep_mid_acc(med_out)
                    

                    ans_final = batch.select_idxs(oqa_train_idxs) if len(oqa_train_idxs) > 0 else None
                    def tag_source(dp: DataProto, tag: str) -> DataProto:
                        if dp is None:
                            return None
                        n = len(dp)
                        dp.non_tensor_batch["clpo_source"] = np.array([tag] * n, dtype=object)
                        return dp

                    ans_final = tag_source(ans_final, "oqa") if ans_final is not None else None
                    hard_keep = tag_source(hard_keep, "rewritten_hard")
                    med_keep = tag_source(med_keep, "rewritten_medium")

                    to_mix = [x for x in [ans_final, hard_keep, med_keep] if x is not None]
                    if not to_mix:
                        mixed = batch  # Fallback to full batch
                    else:
                        try:
                            src_names = []
                            src_keys = []
                            for dp in [ans_final, hard_keep, med_keep]:
                                if dp is None:
                                    src_names.append("none")
                                    src_keys.append(set())
                                else:
                                    src_names.append(str(dp.non_tensor_batch.get("clpo_source", np.array(["unknown"]))[0]))
                                    src_keys.append(set(dp.non_tensor_batch.keys()))

                            print("[CLPO DEBUG] non_tensor_batch keys before concat:")
                            for name, keys in zip(src_names, src_keys):
                                print(f"  - source={name}: {sorted(list(keys))}")

                            all_keys = set()
                            for dp in to_mix:
                                all_keys.update(dp.non_tensor_batch.keys())

                            for dp in to_mix:
                                missing = [k for k in all_keys if k not in dp.non_tensor_batch]
                                if missing:
                                    print(f"[CLPO DEBUG] filling missing keys for {dp.non_tensor_batch.get('clpo_source', np.array(['unknown']))[0]}: {missing}")
                                for k in missing:
                                    dp.non_tensor_batch[k] = np.array([None] * len(dp), dtype=object)

                        except Exception as e:
                            print(f"[CLPO DEBUG] key alignment pre-concat failed with error: {e}")

                        mixed = DataProto.concat(to_mix)

                    try:
                        mixed.batch["response_mask"] = compute_response_mask(mixed)
                    except Exception:
                        pass

                    world_size = self.actor_rollout_wg.world_size
                    ori_len = len(mixed)
                    keep_len = ori_len - (ori_len % world_size)
                    
                    if keep_len <= 0:
                        print("Batch too small for world_size, skipping iteration")
                        progress_bar.update(1)
                        self.global_steps += 1
                        continue

                    hard_tag = set()
                    if hard_keep is not None:
                        # Use response text hashing to identify hard samples
                        hard_response_hashes = set()
                        for i in range(len(hard_keep)):
                            try:
                                resp_text = self.tokenizer.decode(hard_keep.batch["responses"][i], skip_special_tokens=True)
                                hard_response_hashes.add(hash(resp_text.strip()))
                            except Exception:
                                continue
                        
                        for i in range(len(mixed)):
                            try:
                                resp_text = self.tokenizer.decode(mixed.batch["responses"][i], skip_special_tokens=True)
                                if hash(resp_text.strip()) in hard_response_hashes:
                                    hard_tag.add(i)
                            except Exception:
                                continue

                    idxs_all = list(range(len(mixed)))
                    hard_first = [i for i in idxs_all if i in hard_tag]
                    others = [i for i in idxs_all if i not in hard_tag]
                    keep_indices = (hard_first + others)[:keep_len]
                    mixed = mixed.select_idxs(keep_indices)

                    total_sample_count = len(mixed)
                    metrics["clpo/final_batch/total_sample_count"] = total_sample_count
                    
                    src = mixed.non_tensor_batch.get("clpo_source", np.array(["unknown"] * len(mixed), dtype=object))
                    uids_final = mixed.non_tensor_batch["uid"]
                    
                    def _count_src(label: str) -> int:
                        return int(np.sum(src == label))
                    
                    cnt_oqa = _count_src("oqa")
                    cnt_rh = _count_src("rewritten_hard") if should_rewrite_hard else 0
                    cnt_rm = _count_src("rewritten_medium") if should_rewrite_medium else 0
                    
                    from_orig_medium_oqa_count = 0
                    from_orig_medium_rewritten_count = 0
                    from_orig_hard_oqa_count = 0
                    from_orig_hard_rewritten_count = 0
                    
                    for i, uid_final in enumerate(uids_final):
                        source = src[i]
                        original_acc = uid2acc.get(uid_final, -1)  
                        
                        is_orig_medium = (self.clpo_med_acc_lower < original_acc <= self.clpo_med_acc_upper)
                        is_orig_hard = (0.0 <= original_acc <= self.clpo_hard_acc_upper)
                        
                        if source == "oqa":
                            if is_orig_medium:
                                from_orig_medium_oqa_count += 1
                            elif is_orig_hard:
                                from_orig_hard_oqa_count += 1
                        elif source == "rewritten_medium" and is_orig_medium:
                            from_orig_medium_rewritten_count += 1
                        elif source == "rewritten_hard" and is_orig_hard:
                            from_orig_hard_rewritten_count += 1
                    
                    metrics["clpo/final_batch/source/from_orig_medium/oqa_sample_count"] = from_orig_medium_oqa_count
                    metrics["clpo/final_batch/source/from_orig_medium/rewritten_sample_count"] = from_orig_medium_rewritten_count
                    metrics["clpo/final_batch/source/from_orig_hard/oqa_sample_count"] = from_orig_hard_oqa_count
                    metrics["clpo/final_batch/source/from_orig_hard/rewritten_sample_count"] = from_orig_hard_rewritten_count
                    
                    def _ratio(x: int) -> float:
                        return round(100.0 * x / max(1, total_sample_count), 2)
                    
                    total_from_orig_medium = from_orig_medium_oqa_count + from_orig_medium_rewritten_count
                    total_from_orig_hard = from_orig_hard_oqa_count + from_orig_hard_rewritten_count
                    
                    metrics["clpo/final_batch/source_ratio/from_orig_medium_pct"] = _ratio(total_from_orig_medium)
                    metrics["clpo/final_batch/source_ratio/from_orig_hard_pct"] = _ratio(total_from_orig_hard)

                    metrics["clpo/ablation/rewrite_effectiveness/hard_rewrite_count"] = cnt_rh
                    metrics["clpo/ablation/rewrite_effectiveness/medium_rewrite_count"] = cnt_rm
                    metrics["clpo/ablation/rewrite_effectiveness/total_rewrite_count"] = cnt_rh + cnt_rm
                    metrics["clpo/ablation/rewrite_effectiveness/rewrite_ratio"] = _ratio(cnt_rh + cnt_rm)
                    
                    
                    if self.clpo_rewrite_mode == "hard_only":
                        metrics["clpo/ablation/comparison/rewrite_type"] = "hard_only"
                        metrics["clpo/ablation/comparison/medium_rewrite_disabled"] = 1
                    elif self.clpo_rewrite_mode == "medium_only":
                        metrics["clpo/ablation/comparison/rewrite_type"] = "medium_only"
                        metrics["clpo/ablation/comparison/hard_rewrite_disabled"] = 1
                    else:  # both
                        metrics["clpo/ablation/comparison/rewrite_type"] = "both"
                        metrics["clpo/ablation/comparison/hard_rewrite_enabled"] = 1
                        metrics["clpo/ablation/comparison/medium_rewrite_enabled"] = 1
                    
                    
                    initial_fully_wrong_hard_count = 0
                    rescued_via_rewrite_count = 0
                    
                    
                    fully_wrong_hard_uids = []
                    for uid in hard_problem_uids:
                        if uid2acc[uid] == 0.0:
                            initial_fully_wrong_hard_count += 1
                            fully_wrong_hard_uids.append(uid)
                    
                    
                    if hard_out is not None and len(hard_out) > 0:
                        seq_rewards = hard_out.batch["reward_tensor"].sum(dim=-1).cpu().numpy()
                        acc_arr = (seq_rewards > 0).astype(float)
                        uids = hard_out.non_tensor_batch["uid"]
                        
                        uid2rewrite_acc = {}
                        for i, u in enumerate(uids):
                            uid2rewrite_acc.setdefault(u, []).append(float(acc_arr[i]))
                        uid2rewrite_acc = {u: float(np.mean(v)) for u, v in uid2rewrite_acc.items()}
                        
                        
                        for uid in fully_wrong_hard_uids:
                            if uid in uid2rewrite_acc:
                                rewrite_acc = uid2rewrite_acc[uid]
                                if 0.0 < rewrite_acc < 1.0:  
                                    rescued_via_rewrite_count += 1
                    
                    
                    metrics["clpo/salvage/initial_fully_wrong_hard_problem_count"] = initial_fully_wrong_hard_count
                    metrics["clpo/salvage/rescued_via_rewrite_problem_count"] = rescued_via_rewrite_count if should_rewrite_hard else 0
                    
                    
                    rescue_rate = 0.0
                    if initial_fully_wrong_hard_count > 0 and should_rewrite_hard:
                        rescue_rate = round(100.0 * rescued_via_rewrite_count / initial_fully_wrong_hard_count, 2)
                    metrics["clpo/salvage/rescue_rate_pct"] = rescue_rate
                    
                    
                    unique_problems_in_final_batch = set(uids_final)
                    
                    
                    hard_problems_in_final_batch_count = 0
                    medium_problems_in_final_batch_count = 0
                    
                    for uid in unique_problems_in_final_batch:
                        original_acc = uid2acc.get(uid, -1)
                        if 0.0 <= original_acc <= self.clpo_hard_acc_upper:
                            hard_problems_in_final_batch_count += 1
                        elif self.clpo_med_acc_lower < original_acc <= self.clpo_med_acc_upper:
                            medium_problems_in_final_batch_count += 1
                    
                    
                    metrics["clpo/injection/hard_problems_in_final_batch_count"] = hard_problems_in_final_batch_count
                    metrics["clpo/injection/medium_problems_in_final_batch_count"] = medium_problems_in_final_batch_count
                    

                    hard_problem_inclusion_rate = 0.0
                    if hard_problem_count > 0:
                        hard_problem_inclusion_rate = round(100.0 * hard_problems_in_final_batch_count / hard_problem_count, 2)
                    metrics["clpo/injection/hard_problem_inclusion_rate_pct"] = hard_problem_inclusion_rate
                    
                    medium_problem_inclusion_rate = 0.0
                    if medium_problem_count > 0:
                        medium_problem_inclusion_rate = round(100.0 * medium_problems_in_final_batch_count / medium_problem_count, 2)
                    metrics["clpo/injection/medium_problem_inclusion_rate_pct"] = medium_problem_inclusion_rate
                    
                    
                    if self.clpo_rewrite_mode == "hard_only":
                        print(
                            f"[CLPO] Final mix (count | ratio%) - HARD ONLY: "
                            f"OQA={cnt_oqa}({_ratio(cnt_oqa)}%), "
                            f"Rew-H={cnt_rh}({_ratio(cnt_rh)}%), "
                            f"Total={total_sample_count}"
                        )
                    elif self.clpo_rewrite_mode == "medium_only":
                        print(
                            f"[CLPO] Final mix (count | ratio%) - MEDIUM ONLY: "
                            f"OQA={cnt_oqa}({_ratio(cnt_oqa)}%), "
                            f"Rew-M={cnt_rm}({_ratio(cnt_rm)}%), "
                            f"Total={total_sample_count}"
                        )
                    else:  
                        print(
                            f"[CLPO] Final mix (count | ratio%) - BOTH: "
                            f"OQA={cnt_oqa}({_ratio(cnt_oqa)}%), "
                            f"Rew-H={cnt_rh}({_ratio(cnt_rh)}%), "
                            f"Rew-M={cnt_rm}({_ratio(cnt_rm)}%), Total={total_sample_count}"
                        )
                    
                    if should_rewrite_hard:
                        print(
                            f"[CLPO] Core Impact Analysis: "
                            f"Hard Rescue Rate={rescue_rate}% ({rescued_via_rewrite_count}/{initial_fully_wrong_hard_count}), "
                            f"Hard Inclusion Rate={hard_problem_inclusion_rate}% ({hard_problems_in_final_batch_count}/{hard_problem_count})"
                        )
                    else:
                        print(f"[CLPO] Hard rewrite disabled - no rescue analysis")

                    easy_uids = [u for u, a in uid2acc.items() if a > self.clpo_med_acc_upper]
                    ori_hard = len([u for u in uid if u in hard_uids])
                    ori_med = len([u for u in uid if u in med_uids])
                    ori_easy = len([u for u in uid if u in easy_uids])
                    print(
                        f"[CLPO] Original difficulty counts (rollout items): "
                        f"easy={ori_easy}, medium={ori_med}, hard={ori_hard}"
                    )

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(mixed)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = mixed.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        metrics.update({"actor/entropy": entropy_agg.detach().item()})
                        old_log_prob.batch.pop("entropys")
                        mixed = mixed.union(old_log_prob)

                        if "rollout_log_probs" in mixed.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(mixed))

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(mixed)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(mixed)
                            mixed = mixed.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(mixed)
                            mixed = mixed.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            # The original future_reward was computed on the full batch, but we now have a mixed batch
                            # We need to recompute rewards for the mixed batch
                            reward_tensor, reward_extra_infos_dict = compute_reward(mixed, self.reward_fn)
                        else:
                            # For non-async case, we also need to recompute for mixed batch
                            reward_tensor, reward_extra_infos_dict = compute_reward(mixed, self.reward_fn)
                        
                        mixed.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            mixed.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        # Prepare difficulty-conditioned scaling vectors for KL
                        try:
                            src = mixed.non_tensor_batch.get(
                                "clpo_source", np.array(["oqa"] * len(mixed), dtype=object)
                            )
                            
                            # For fine-grained difficulty classification, we need to check accuracy for OQA samples
                            uid = mixed.non_tensor_batch["uid"]
                            if "acc" in mixed.batch.keys():
                                acc_arr = np.asarray([x.item() if torch.is_tensor(x) else float(x) for x in mixed.batch["acc"]], dtype=float)
                            elif "acc" in mixed.non_tensor_batch.keys():
                                acc_arr = np.asarray(mixed.non_tensor_batch["acc"], dtype=float)
                            else:
                                # Fallback: reward > 0 => correct
                                seq_rewards = mixed.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                                acc_arr = (seq_rewards > 0).astype(float)

                            uid2acc = {}
                            for i, u in enumerate(uid):
                                uid2acc.setdefault(u, []).append(float(acc_arr[i]))
                            uid2acc = {u: float(np.mean(v)) for u, v in uid2acc.items()}
                            
                            # Create difficulty labels based on both source and accuracy
                            difficulty_labels = []
                            for i, s in enumerate(src):
                                if str(s) == "rewritten_hard":
                                    difficulty_labels.append("hard")
                                elif str(s) == "rewritten_medium":
                                    difficulty_labels.append("medium")
                                elif str(s) == "oqa":
                                    # Further classify OQA samples by accuracy
                                    u = uid[i]
                                    acc = uid2acc.get(u, 0.5)  # Default to medium if not found
                                    if acc <= self.clpo_hard_acc_upper:
                                        difficulty_labels.append("hard")
                                    elif self.clpo_med_acc_lower < acc <= self.clpo_med_acc_upper:
                                        difficulty_labels.append("medium")
                                    else:
                                        difficulty_labels.append("easy")
                                else:
                                    difficulty_labels.append("medium")  # Default fallback
                            
                            # Set difficulty_source for KL metrics recording
                            mixed.non_tensor_batch["difficulty_source"] = np.array(difficulty_labels, dtype=object)
                            
                            # In-reward multipliers based on fine-grained difficulty
                            r_hard = float(self.config.algorithm.get("kl_in_reward_coef_hard", 1.0))
                            r_non = float(self.config.algorithm.get("kl_in_reward_coef_nonhard", 1.0))
                            r_scale = np.array([
                                r_hard if str(d) == "hard" else r_non for d in difficulty_labels
                            ], dtype=float)
                            mixed.non_tensor_batch["kl_in_reward_scale"] = r_scale

                            # In-loss multipliers based on fine-grained difficulty
                            l_hard = float(self.config.actor_rollout_ref.actor.get("kl_loss_coef_hard_scale", 1.0))
                            l_non = float(self.config.actor_rollout_ref.actor.get("kl_loss_coef_nonhard_scale", 1.0))
                            l_scale = torch.tensor([
                                l_hard if str(d) == "hard" else l_non for d in difficulty_labels
                            ], dtype=mixed.batch["token_level_scores"].dtype, device=mixed.batch["token_level_scores"].device)
                            mixed.batch["kl_in_loss_scale"] = l_scale
                        except Exception as _e:
                            print(f"[DCKL] scale preparation failed: {_e}")

                        if self.config.algorithm.use_kl_in_reward:
                            mixed, kl_metrics = apply_kl_penalty(
                                mixed, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            mixed.batch["token_level_rewards"] = mixed.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        mixed = compute_advantage(
                            mixed,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(mixed)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            mixed.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(mixed)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                                        # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(mixed.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(mixed.batch["responses"], skip_special_tokens=True)
                            scores = mixed.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in mixed
                            ]

                            if "request_id" in mixed.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    mixed.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # Validation and checkpointing
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    
                    # Check if this is the best model and save if needed
                    if self._is_best_model(val_metrics):
                        with marked_timer("save_best_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                            print("Saved best model checkpoint")

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=mixed, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=mixed, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=mixed, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1


                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return



    def _add_rewrite_data_to_buffer(self, original_questions: list[str], original_responses: list[str], 
                                  rewritten_questions: list[str], rewritten_responses: list[str], difficulty: str = "unknown"):
        """Add rewrite data to buffer for later saving"""
        if not self.clpo_save_rewrite_data:
            return
            
        # Ensure all lists have the same length
        min_len = min(len(original_questions), len(original_responses), 
                     len(rewritten_questions), len(rewritten_responses))
        
        for i in range(min_len):
            rewrite_entry = {
                "ORIGINAL_QUESTION": original_questions[i],
                "ORIGINAL_RESPONSE": original_responses[i], 
                "REWRITTEN_QUESTION": rewritten_questions[i],
                "REWRITTEN_RESPONSE": rewritten_responses[i],
                "DIFFICULTY": difficulty
            }
            
            # Add to general buffer
            self.rewrite_data_buffer.append(rewrite_entry)
            
            # Add to specific difficulty buffer
            if difficulty == "hard":
                self.hard_rewrite_data_buffer.append(rewrite_entry)
            elif difficulty == "medium":
                self.medium_rewrite_data_buffer.append(rewrite_entry)

    def _save_all_rewrite_data(self):
        """Save all rewrite data to files"""
        if not self.clpo_save_rewrite_data:
            return
            
        # Save combined data if path is specified
        if self.clpo_rewrite_save_path and self.rewrite_data_buffer:
            self._save_rewrite_data_to_file(self.rewrite_data_buffer, self.clpo_rewrite_save_path)
        
        # Save hard data if path is specified
        if self.clpo_hard_rewrite_save_path and self.hard_rewrite_data_buffer:
            self._save_rewrite_data_to_file(self.hard_rewrite_data_buffer, self.clpo_hard_rewrite_save_path)
        
        # Save medium data if path is specified
        if self.clpo_medium_rewrite_save_path and self.medium_rewrite_data_buffer:
            self._save_rewrite_data_to_file(self.medium_rewrite_data_buffer, self.clpo_medium_rewrite_save_path)

    def _save_rewrite_data_to_file(self, data_buffer: list, save_path: str):
        """Save rewrite data to JSON file"""
        if not save_path:
            print("[WARNING] CLPO rewrite save path not specified, skipping save")
            return
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save to JSON file
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data_buffer, f, ensure_ascii=False, indent=2)
            
            print(f"[INFO] Saved {len(data_buffer)} rewrite samples to {save_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save rewrite data to {save_path}: {e}")

    def _save_rewrite_data_step(self):
        """Save rewrite data for current step (incremental save)"""
        if not self.clpo_save_rewrite_data:
            return
            
        step = self.global_steps
        
        # Save combined data if path is specified (both step file and final file)
        if self.clpo_rewrite_save_path and self.rewrite_data_buffer:
            # Create step-specific filename
            base_path = self.clpo_rewrite_save_path
            if base_path.endswith('.json'):
                step_path = base_path.replace('.json', f'_step_{step}.json')
            else:
                step_path = f"{base_path}_step_{step}.json"
            # Save step file
            self._save_rewrite_data_to_file(self.rewrite_data_buffer, step_path)
            # Also update final file
            self._save_rewrite_data_to_file(self.rewrite_data_buffer, base_path)
        
        # Save hard data if path is specified (both step file and final file)
        if self.clpo_hard_rewrite_save_path and self.hard_rewrite_data_buffer:
            base_path = self.clpo_hard_rewrite_save_path
            if base_path.endswith('.json'):
                step_path = base_path.replace('.json', f'_step_{step}.json')
            else:
                step_path = f"{base_path}_step_{step}.json"
            # Save step file
            self._save_rewrite_data_to_file(self.hard_rewrite_data_buffer, step_path)
            # Also update final file
            self._save_rewrite_data_to_file(self.hard_rewrite_data_buffer, base_path)
        
        # Save medium data if path is specified (both step file and final file)
        if self.clpo_medium_rewrite_save_path and self.medium_rewrite_data_buffer:
            base_path = self.clpo_medium_rewrite_save_path
            if base_path.endswith('.json'):
                step_path = base_path.replace('.json', f'_step_{step}.json')
            else:
                step_path = f"{base_path}_step_{step}.json"
            # Save step file
            self._save_rewrite_data_to_file(self.medium_rewrite_data_buffer, step_path)
            # Also update final file
            self._save_rewrite_data_to_file(self.medium_rewrite_data_buffer, base_path)
