import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import get_linear_schedule_with_warmup
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import argparse
import os
from pathlib import Path
import logging
from tqdm import tqdm
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QSFTTransition:
    state: str
    action: str
    reward: float
    next_state: str
    task_description: str
    instance_id: str
    step_number: int
    total_steps: int
    task_completed: bool

class QSFTDataset(Dataset):
    
    def __init__(self, transitions: List[QSFTTransition], tokenizer, max_length: int = 512):
        self.transitions = transitions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        transition = self.transitions[idx]
        
        state_encoding = self.tokenizer(
            f"State: {transition.state}",
            truncation=True,
            max_length=self.max_length // 2,
            padding='max_length',
            return_tensors='pt'
        )
        
        next_state_encoding = self.tokenizer(
            f"State: {transition.next_state}",
            truncation=True,
            max_length=self.max_length // 2,
            padding='max_length',
            return_tensors='pt'
        )
        
        action_encoding = self.tokenizer(
            transition.action,
            truncation=True,
            max_length=50,  
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'state_input_ids': state_encoding['input_ids'].squeeze(),
            'state_attention_mask': state_encoding['attention_mask'].squeeze(),
            'next_state_input_ids': next_state_encoding['input_ids'].squeeze(),
            'next_state_attention_mask': next_state_encoding['attention_mask'].squeeze(),
            'action_input_ids': action_encoding['input_ids'].squeeze(),
            'action_attention_mask': action_encoding['attention_mask'].squeeze(),
            'reward': torch.tensor(transition.reward, dtype=torch.float),
            'task_completed': torch.tensor(transition.task_completed, dtype=torch.bool)
        }

class QSFTLoss(nn.Module):
    def __init__(self, gamma: float = 0.99):
        super().__init__()
        self.gamma = gamma

    def compute_bellman_targets(self, rewards, target_next_logits, behavior_next_logits, next_state_mask):
        """
        Compute Bellman targets: B*pθ̄(a|s) = r + γ max_a' pθ̄(a'|s')/πβ(a'|s')
        
        Args:
            rewards: [B] - immediate rewards
            target_next_logits: [B, T, V] - logits from target Q-model for next states
            behavior_next_logits: [B, T, V] - logits from behavior model for next states  
            next_state_mask: [B, T] - attention mask for next states
            
        Returns:
            bellman_targets: [B] - Bellman backup values
        """
        target_next_probs = F.softmax(target_next_logits[:, -1, :], dim=-1)  # [B, V] - last token probs
        behavior_next_probs = F.softmax(behavior_next_logits[:, -1, :], dim=-1)  # [B, V]

        q_over_behavior = target_next_probs / (behavior_next_probs + 1e-8)  # [B, V]
        
        max_q_over_behavior, _ = torch.max(q_over_behavior, dim=-1)  # [B]
        
        bellman_targets = rewards + self.gamma * max_q_over_behavior  # [B]
        
        bellman_targets = torch.clamp(bellman_targets, min=0.0, max=1.0)
        
        return bellman_targets

    def forward(self, logits, action_ids, bellman_backup, action_token_pos):
        B, T, V = logits.shape

        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        
        action_pos = action_token_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)  # [B, 1, V]
        action_log_probs = log_probs.gather(1, action_pos).squeeze(1)  # [B, V]
        
        target_tokens = action_ids.gather(1, action_token_pos.unsqueeze(1)).squeeze(1)  # [B]
        
        actual_action_log_prob = action_log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)  # [B]
        first_term = bellman_backup * actual_action_log_prob  # [B]
        
        vocab_size = V
        other_actions_weight = (1 - bellman_backup) / (vocab_size - 1)  # [B]
        
        all_actions_log_prob_sum = action_log_probs.sum(dim=-1)  # [B]
        
        other_actions_log_prob_sum = all_actions_log_prob_sum - actual_action_log_prob  # [B]
        
        second_term = other_actions_weight * other_actions_log_prob_sum  # [B]
        
        total_loss = -(first_term + second_term)  # [B]
        
        return total_loss.mean()

class QSFTTrainer:
    
    def __init__(self, model_name: str = "gpt2-medium", device: str = None, 
                 gamma: float = 0.99, beta: float = 1.0, target_update_rate: float = 0.005):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.gamma = gamma
        self.beta = beta
        self.target_update_rate = target_update_rate
        
        logger.info(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.behavior_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.behavior_model.to(self.device)
        
        self.q_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.q_model.to(self.device)
        
        self.target_q_model = copy.deepcopy(self.q_model)
        self.target_q_model.to(self.device)
        
        self.qsft_loss = QSFTLoss(gamma=self.gamma)
        
        logger.info(f"Models loaded on {self.device}")
    
    def load_dataset(self, dataset_path: str) -> List[QSFTTransition]:
        logger.info(f"Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        transitions = []
        for item in data:
            transition = QSFTTransition(
                state=item['state'],
                action=item['action'],
                reward=item['reward'],
                next_state=item['next_state'],
                task_description=item['task_description'],
                instance_id=item['instance_id'],
                step_number=item['step_number'],
                total_steps=item['total_steps'],
                task_completed=item['task_completed']
            )
            transitions.append(transition)
        
        logger.info(f"Loaded {len(transitions)} transitions")
        return transitions
    
    def create_dataloader(self, transitions: List[QSFTTransition], 
                         batch_size: int = 4, max_length: int = 512) -> DataLoader:     
        dataset = QSFTDataset(transitions, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    
    def train_behavior_policy(self, dataloader: DataLoader, 
                            num_epochs: int = 3,
                            learning_rate: float = 5e-5) -> None:
        logger.info("Training behavior policy π_φ...")
        
        optimizer = optim.AdamW(self.behavior_model.parameters(), lr=learning_rate)
        self.behavior_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Behavior Policy Epoch {epoch + 1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                state_ids = batch['state_input_ids'].to(self.device)
                action_ids = batch['action_input_ids'].to(self.device)
                action_mask = batch['action_attention_mask'].to(self.device)
                
                full_input = torch.cat([state_ids, action_ids], dim=1)
                
                labels = full_input.clone()
                state_len = state_ids.shape[1]
                labels[:, :state_len] = -100
                
                outputs = self.behavior_model(input_ids=full_input, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Behavior policy epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    def train_q_values(self, dataloader: DataLoader,
                      num_epochs: int = 3,
                      learning_rate: float = 5e-5,
                      warmup_steps: int = 100,
                      save_steps: int = 500,
                      output_dir: str = "qsft_model") -> None:
        logger.info("Training Q-value model p_θ...")

        os.makedirs(output_dir, exist_ok=True)

        optimizer = optim.AdamW(self.q_model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.behavior_model.eval()
        self.target_q_model.eval()

        global_step = 0

        for epoch in range(num_epochs):
            logger.info(f"Q-value training epoch {epoch + 1}/{num_epochs}")
            self.q_model.train()
            total_loss = 0

            progress_bar = tqdm(dataloader, desc=f"Q-value Epoch {epoch + 1}")

            for batch in progress_bar:
                state_ids = batch['state_input_ids'].to(self.device)
                next_state_ids = batch['next_state_input_ids'].to(self.device)
                next_state_mask = batch['next_state_attention_mask'].to(self.device)
                action_ids = batch['action_input_ids'].to(self.device)
                action_mask = batch['action_attention_mask'].to(self.device)
                rewards = batch['reward'].to(self.device)

                current_input = torch.cat([state_ids, action_ids], dim=1)
                current_outputs = self.q_model(input_ids=current_input)
                current_logits = current_outputs.logits

                with torch.no_grad():
                    target_outputs = self.target_q_model(input_ids=next_state_ids, attention_mask=next_state_mask)
                    target_next_logits = target_outputs.logits
                    
                    behavior_outputs = self.behavior_model(input_ids=next_state_ids, attention_mask=next_state_mask)
                    behavior_next_logits = behavior_outputs.logits

                    bellman_weights = self.qsft_loss.compute_bellman_targets(
                        rewards, target_next_logits, behavior_next_logits, next_state_mask
                    )

                action_token_pos = torch.full(
                    (state_ids.size(0),),
                    fill_value=state_ids.size(1),
                    dtype=torch.long,
                    device=state_ids.device
                )

                loss = self.qsft_loss(
                    logits=current_logits,
                    action_ids=current_input,
                    bellman_backup=bellman_weights,
                    action_token_pos=action_token_pos
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                self._update_target_network()

                total_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (global_step % len(dataloader) + 1):.4f}',
                    'avg_bellman': f'{bellman_weights.mean().item():.4f}'
                })

                if global_step % save_steps == 0:
                    self.save_checkpoint(output_dir, global_step)

            self.save_checkpoint(output_dir, f"epoch_{epoch + 1}")

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Q-value epoch {epoch + 1} average loss: {avg_loss:.4f}")

        self.save_checkpoint(output_dir, "final")
        logger.info("Q-value training completed!")
    
    def _update_target_network(self):
        for target_param, param in zip(self.target_q_model.parameters(), self.q_model.parameters()):
            target_param.data.copy_(
                (1 - self.target_update_rate) * target_param.data + 
                self.target_update_rate * param.data
            )
    
    def train(self, dataloader: DataLoader, 
              num_epochs: int = 3,
              learning_rate: float = 5e-5,
              warmup_steps: int = 100,
              save_steps: int = 500,
              output_dir: str = "qsft_model"):
        
        self.train_behavior_policy(dataloader, num_epochs=2, learning_rate=learning_rate)
        
        self.train_q_values(
            dataloader, 
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            output_dir=output_dir
        )
    
    def extract_policy_probabilities(self, state: str, max_length: int = 100) -> torch.Tensor:
        state_input = self.tokenizer(
            f"State: {state}",
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        self.behavior_model.eval()
        self.q_model.eval()
        
        with torch.no_grad():
            behavior_outputs = self.behavior_model(**state_input)
            behavior_logits = behavior_outputs.logits[0, -1, :]  
            behavior_probs = F.softmax(behavior_logits, dim=-1)
            
            q_outputs = self.q_model(**state_input)
            q_logits = q_outputs.logits[0, -1, :]  
            q_probs = F.softmax(q_logits, dim=-1)
            
            combined_logits = torch.log(behavior_probs + 1e-8) + self.beta * torch.log(q_probs + 1e-8)
            final_probs = F.softmax(combined_logits, dim=-1)
        
        return final_probs
    
    def generate_action(self, state: str, max_length: int = 100) -> str:
        action_probs = self.extract_policy_probabilities(state, max_length)
        
        action_token_id = torch.multinomial(action_probs, 1).item()
        
        action = self.tokenizer.decode([action_token_id], skip_special_tokens=True)
        
        return action
    
    def save_checkpoint(self, output_dir: str, step: str):
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Only save the Q-value model during training
        self.q_model.save_pretrained(os.path.join(checkpoint_dir, "q_model"))
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save behavior model reference (not the full model)
        behavior_path = os.path.join(output_dir, "behavior_only")
        if os.path.exists(behavior_path):
            with open(os.path.join(checkpoint_dir, "behavior_model_path.txt"), 'w') as f:
                f.write(behavior_path)
        
        # Save config to reconstruct target model
        config = {
            'gamma': self.gamma,
            'beta': self.beta,
            'target_update_rate': self.target_update_rate,
            'model_name': self.model_name,
            'reconstruct_target': True  # Flag to rebuild target from q_model
        }
        
        with open(os.path.join(checkpoint_dir, 'qsft_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.q_model.eval()
        self.behavior_model.eval()
        
        total_loss = 0
        total_reward = 0
        completed_tasks = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                state_ids = batch['state_input_ids'].to(self.device)
                action_ids = batch['action_input_ids'].to(self.device)
                rewards = batch['reward'].to(self.device)
                task_completed = batch['task_completed'].to(self.device)
                
                full_input = torch.cat([state_ids, action_ids], dim=1)
                labels = full_input.clone()
                state_len = state_ids.shape[1]
                labels[:, :state_len] = -100
                
                outputs = self.behavior_model(input_ids=full_input, labels=labels)
                total_loss += outputs.loss.item()
                
                total_reward += rewards.sum().item()
                completed_tasks += task_completed.sum().item()
                total_samples += len(rewards)
        
        avg_loss = total_loss / len(dataloader)
        avg_reward = total_reward / total_samples
        completion_rate = completed_tasks / total_samples
        
        return {
            'avg_loss': avg_loss,
            'avg_reward': avg_reward,
            'completion_rate': completion_rate
        }

def main():
    parser = argparse.ArgumentParser(description="Q-SFT Training for GPT-2 Medium")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the Q-SFT dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="qsft_model",
                       help="Directory to save the trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_split", type=float, default=0.1,
                       help="Fraction of data to use for evaluation")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Policy extraction hyperparameter")
    
    args = parser.parse_args()
    
    trainer = QSFTTrainer(gamma=args.gamma, beta=args.beta)
    
    transitions = trainer.load_dataset(args.dataset_path)
    
    split_idx = int(len(transitions) * (1 - args.eval_split))
    train_transitions = transitions[:split_idx]
    eval_transitions = transitions[split_idx:]
    
    logger.info(f"Train set: {len(train_transitions)} transitions")
    logger.info(f"Eval set: {len(eval_transitions)} transitions")
    
    train_dataloader = trainer.create_dataloader(
        train_transitions, args.batch_size, args.max_length
    )
    eval_dataloader = trainer.create_dataloader(
        eval_transitions, args.batch_size, args.max_length
    )
    
    trainer.train(
        train_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        output_dir=args.output_dir
    )
    
    logger.info("Evaluating model...")
    eval_metrics = trainer.evaluate(eval_dataloader)
    
    logger.info("Evaluation Results:")
    for metric, value in eval_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()