# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.workers.config import ActorConfig


def sft_loss(config: ActorConfig, model_output, data, dp_group=None):
    log_prob = model_output["log_probs"]  # [bsz, response_length]
    response_mask = data["response_mask"].to(bool)
    loss = -torch.mean(log_prob * response_mask)
    return loss, {"loss": loss.detach().item()}


def ppo_loss(config: ActorConfig, model_output, data, dp_group=None):
    log_prob = model_output["log_probs"]
    entropy = model_output.get("entropy", None)

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
    )

    metrics.update(
        {
            "pg_loss": pg_loss.detach().item(),
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        }
    )
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        # Optional per-sample scaling for KL-in-loss
        if "kl_in_loss_scale" in data:
            scale = data["kl_in_loss_scale"]
            if not torch.is_tensor(scale):
                scale = torch.as_tensor(scale, device=kld.device, dtype=kld.dtype)
            kld = kld * scale.unsqueeze(-1)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef
        
        # Add separate metrics for hard and non-hard samples if difficulty classification is available
        if "difficulty_source" in data:
            difficulty_source = data["difficulty_source"]
            import numpy as np
            
            # Convert to numpy array if needed
            if hasattr(difficulty_source, 'cpu'):
                difficulty_source = difficulty_source.cpu().numpy()
            elif not isinstance(difficulty_source, np.ndarray):
                difficulty_source = np.array(difficulty_source)
            
            # Calculate original KL loss (without scaling) for all samples
            kld_original = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
            kl_loss_original = agg_loss(loss_mat=kld_original, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)
            metrics["actor/kl_loss_original"] = kl_loss_original.detach().item()
            
            # Separate hard and non-hard samples
            hard_mask = np.array([str(s) == "hard" for s in difficulty_source])
            non_hard_mask = ~hard_mask
            
            if np.any(hard_mask):
                hard_mask_tensor = torch.from_numpy(hard_mask).to(kld.device)
                hard_response_mask = response_mask[hard_mask_tensor]
                if hard_response_mask.sum() > 0:
                    # Calculate hard samples KL loss (original, without scaling)
                    hard_kld_original = kld_original[hard_mask_tensor]
                    hard_kl_loss_original = agg_loss(loss_mat=hard_kld_original, loss_mask=hard_response_mask, loss_agg_mode=config.loss_agg_mode)
                    metrics["actor/kl_loss_hard"] = hard_kl_loss_original.detach().item()
                    
                    # Calculate hard samples KL loss (scaled)
                    hard_kld_scaled = kld[hard_mask_tensor]
                    hard_kl_loss_scaled = agg_loss(loss_mat=hard_kld_scaled, loss_mask=hard_response_mask, loss_agg_mode=config.loss_agg_mode)
                    metrics["actor/kl_loss_hard_scaled"] = hard_kl_loss_scaled.detach().item()
                    
                    # Calculate scaled coefficient for hard samples
                    if "kl_in_loss_scale" in data:
                        hard_scale = data["kl_in_loss_scale"][hard_mask_tensor]
                        hard_scaled_coeff = (config.kl_loss_coef * torch.mean(hard_scale)).item()
                        metrics["actor/kl_coeff_hard_scaled"] = hard_scaled_coeff
                    else:
                        metrics["actor/kl_coeff_hard_scaled"] = config.kl_loss_coef
            
            if np.any(non_hard_mask):
                non_hard_mask_tensor = torch.from_numpy(non_hard_mask).to(kld.device)
                non_hard_response_mask = response_mask[non_hard_mask_tensor]
                if non_hard_response_mask.sum() > 0:
                    # Calculate non-hard samples KL loss (original, without scaling)
                    non_hard_kld_original = kld_original[non_hard_mask_tensor]
                    non_hard_kl_loss_original = agg_loss(loss_mat=non_hard_kld_original, loss_mask=non_hard_response_mask, loss_agg_mode=config.loss_agg_mode)
                    metrics["actor/kl_loss_non_hard"] = non_hard_kl_loss_original.detach().item()
                    
                    # Calculate non-hard samples KL loss (scaled)
                    non_hard_kld_scaled = kld[non_hard_mask_tensor]
                    non_hard_kl_loss_scaled = agg_loss(loss_mat=non_hard_kld_scaled, loss_mask=non_hard_response_mask, loss_agg_mode=config.loss_agg_mode)
                    metrics["actor/kl_loss_non_hard_scaled"] = non_hard_kl_loss_scaled.detach().item()
                    
                    # Calculate scaled coefficient for non-hard samples
                    if "kl_in_loss_scale" in data:
                        non_hard_scale = data["kl_in_loss_scale"][non_hard_mask_tensor]
                        non_hard_scaled_coeff = (config.kl_loss_coef * torch.mean(non_hard_scale)).item()
                        metrics["actor/kl_coeff_non_hard_scaled"] = non_hard_scaled_coeff
                    else:
                        metrics["actor/kl_coeff_non_hard_scaled"] = config.kl_loss_coef

    return policy_loss, metrics
