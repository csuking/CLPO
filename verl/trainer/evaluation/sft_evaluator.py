# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
SFT训练过程中的评测器，基于PPO的评测逻辑
"""

import re
import torch
from typing import Dict, List, Any
from tensordict import TensorDict


class SFTEvaluator:
    """SFT评测器，支持rollout生成和准确率计算"""
    
    def __init__(self, config: Dict[str, Any], tokenizer, device_name: str):
        self.config = config
        self.tokenizer = tokenizer
        self.device_name = device_name
        
        # 答案提取的正则表达式
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        
        # 生成参数
        self.generation_config = {
            "max_new_tokens": config.get("max_new_tokens", 256),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "do_sample": config.get("do_sample", True),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # 加载评测数据以获取ground truth
        self.eval_data = None
        if hasattr(config, "eval_data_path"):
            import pandas as pd
            self.eval_data = pd.read_parquet(config.eval_data_path)
            print(f"[SFT Evaluator] Loaded {len(self.eval_data)} evaluation samples from {config.eval_data_path}")
    
    def extract_answer(self, text: str) -> str:
        """从生成文本中提取<answer></answer>内容"""
        match = self.answer_pattern.search(text)
        if match:
            return match.group(1).strip()
        return ""
    
    def compute_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """计算准确率"""
        if not predictions or not ground_truths:
            return 0.0
        
        correct = 0
        total = min(len(predictions), len(ground_truths))
        
        for pred, gt in zip(predictions[:total], ground_truths[:total]):
            if pred.strip().lower() == gt.strip().lower():
                correct += 1
                
        return correct / total if total > 0 else 0.0
    
    def generate_responses(self, model, prompts: List[str]) -> List[str]:
        """生成回复，类似PPO的rollout"""
        model.eval()
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.get("max_input_length", 512)
                ).to(self.device_name)
                
                # Generate response
                outputs = model.generate(
                    **inputs,
                    **self.generation_config
                )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
                responses.append(response)
        
        return responses
    
    def evaluate(self, model, val_dataloader, global_step: int) -> Dict[str, float]:
        """评测模型"""
        model.eval()
        
        all_prompts = []
        all_ground_truths = []
        all_generated_responses = []
        
        print(f"[SFT Evaluator] Starting evaluation at step {global_step}")
        
        # 收集prompts和ground truths
        for batch in val_dataloader:
            batch = TensorDict(batch, batch_size=batch.batch_size).to(self.device_name)
            
            # 提取prompts
            if "input_ids" in batch:
                prompts = self.tokenizer.batch_decode(
                    batch["input_ids"], 
                    skip_special_tokens=True
                )
                all_prompts.extend(prompts)
            
            # 提取ground truths - 通过prompt匹配来获取对应的GT
            batch_size = batch["input_ids"].shape[0]
            if self.eval_data is not None:
                batch_gt = []
                for prompt in prompts:
                    # 在评测数据中查找匹配的prompt
                    # 这里使用简单的字符串匹配，可能需要更精确的匹配逻辑
                    matched = False
                    for _, row in self.eval_data.iterrows():
                        if row['prompt'] == prompt:
                            batch_gt.append(str(row['ground_truth']))
                            matched = True
                            break
                    if not matched:
                        batch_gt.append("")  # 如果没找到匹配，使用空字符串
                all_ground_truths.extend(batch_gt)
            else:
                # 如果没有原始数据，使用空字符串
                all_ground_truths.extend([""] * batch_size)
        
        # 生成回复
        print(f"[SFT Evaluator] Generating responses for {len(all_prompts)} prompts...")
        all_generated_responses = self.generate_responses(model, all_prompts)
        
        # 提取答案
        all_answers = [self.extract_answer(resp) for resp in all_generated_responses]
        
        # 计算准确率
        accuracy = self.compute_accuracy(all_answers, all_ground_truths)
        
        print(f"[SFT Evaluator] Evaluation completed. Accuracy: {accuracy:.4f}")
        print(f"[SFT Evaluator] Sample predictions vs ground truths:")
        for i in range(min(3, len(all_answers))):
            print(f"  Pred: '{all_answers[i]}' vs GT: '{all_ground_truths[i]}'")
        
        return {
            "val/generation_accuracy": accuracy,
            "val/total_eval_samples": len(all_prompts),
            "val/step": global_step
        }
