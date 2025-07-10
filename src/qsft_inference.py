import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np

class QSFTInference:   
    def __init__(self, model_path: str, device: str = None, beta: float = 1.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.beta = beta
        
        # Load configuration
        config_path = os.path.join(model_path, 'qsft_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.beta = config.get('beta', beta)
                self.gamma = config.get('gamma', 0.99)
                print(f"Loaded config: beta={self.beta}, gamma={self.gamma}")
        
        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Load behavior policy model (π_φ)
        behavior_model_path = os.path.join(model_path, 'behavior_model')
        behavior_ref_path = os.path.join(model_path, 'behavior_model_path.txt')
        
        if os.path.exists(behavior_model_path):
            print(f"Loading behavior policy from {behavior_model_path}...")
            self.behavior_model = GPT2LMHeadModel.from_pretrained(behavior_model_path)
        elif os.path.exists(behavior_ref_path):
            # Read the reference file to get the actual behavior model path
            with open(behavior_ref_path, 'r') as f:
                actual_behavior_path = f.read().strip()
            print(f"Loading behavior policy from reference: {actual_behavior_path}...")
            self.behavior_model = GPT2LMHeadModel.from_pretrained(actual_behavior_path)
        else:
            print(f"Loading behavior policy from {model_path}...")
            self.behavior_model = GPT2LMHeadModel.from_pretrained(model_path)
        
        self.behavior_model.to(self.device)
        self.behavior_model.eval()
        
        # Load Q-value model (p_θ)
        q_model_path = os.path.join(model_path, 'q_model')
        if os.path.exists(q_model_path):
            print(f"Loading Q-value model from {q_model_path}...")
            self.q_model = GPT2LMHeadModel.from_pretrained(q_model_path)
        else:
            print("Q-value model not found, using behavior model for both (fallback to standard generation)")
            self.q_model = self.behavior_model
        
        self.q_model.to(self.device)
        self.q_model.eval()
        
        print(f"Models loaded on {self.device}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'beta': self.beta,
            'gamma': getattr(self, 'gamma', 0.99),
            'tokenizer_vocab_size': len(self.tokenizer),
            'behavior_model_params': sum(p.numel() for p in self.behavior_model.parameters()),
            'q_model_params': sum(p.numel() for p in self.q_model.parameters()),
            'same_models': self.behavior_model is self.q_model
        }
    
    def get_next_token_probabilities(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            behavior_outputs = self.behavior_model(**inputs)
            behavior_logits = behavior_outputs.logits[0, -1, :]  
            behavior_probs = F.softmax(behavior_logits, dim=-1)
            
            q_outputs = self.q_model(**inputs)
            q_logits = q_outputs.logits[0, -1, :] 
            q_probs = F.softmax(q_logits, dim=-1)
        
        return behavior_probs, q_probs
    
    def extract_policy_probabilities(self, input_text: str) -> torch.Tensor:
        behavior_probs, q_probs = self.get_next_token_probabilities(input_text)
        
        # Apply policy extraction formula
        # π_b(a|s) ∝ π_φ(a|s) exp(β p_θ(a|s))
        # In log space: log π_b(a|s) = log π_φ(a|s) + β log p_θ(a|s)
        combined_logits = (
            torch.log(behavior_probs + 1e-8) + 
            self.beta * torch.log(q_probs + 1e-8)
        )
        
        # Convert back to probabilities
        combined_probs = F.softmax(combined_logits, dim=-1)
        
        return combined_probs
    
    def generate_action(self, state: str, max_length: int = 100, 
                       temperature: float = 1.0, do_sample: bool = True) -> str:
        input_text = f"State: {state}\nAction:"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        generated_tokens = []
        current_input = inputs['input_ids']
        
        for _ in range(max_length):
            current_text = self.tokenizer.decode(current_input[0], skip_special_tokens=True)
            
            combined_probs = self.extract_policy_probabilities(current_text)
            
            if temperature != 1.0:
                combined_probs = combined_probs.pow(1.0 / temperature)
                combined_probs = combined_probs / combined_probs.sum()
            
            if do_sample:
                next_token_id = torch.multinomial(combined_probs, 1).item()
            else:
                next_token_id = torch.argmax(combined_probs).item()
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            next_token = self.tokenizer.decode([next_token_id])
            if '\n' in next_token:
                break
            
            generated_tokens.append(next_token_id)
            
            current_input = torch.cat([
                current_input, 
                torch.tensor([[next_token_id]], device=self.device)
            ], dim=1)
        
        if generated_tokens:
            action = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            action = ""
        
        return action.strip()
    
    def generate_action_beam_search(self, state: str, max_length: int = 100,
                                   num_beams: int = 5, early_stopping: bool = True) -> str:
        input_text = f"State: {state}\nAction:"
        
        candidates = []
        
        for _ in range(num_beams):
            action = self.generate_action(
                state, max_length=max_length, 
                temperature=0.8, do_sample=True
            )
            if action:
                candidate_text = f"{input_text} {action}"
                score = self._score_sequence(candidate_text)
                candidates.append((action, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        else:
            return ""
    
    def _score_sequence(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            behavior_outputs = self.behavior_model(**inputs)
            q_outputs = self.q_model(**inputs)
            
            behavior_logits = behavior_outputs.logits[0, :-1, :]  # All but last
            q_logits = q_outputs.logits[0, :-1, :]
            
            target_ids = inputs['input_ids'][0, 1:]  # All but first
            
            behavior_log_probs = F.log_softmax(behavior_logits, dim=-1)
            q_log_probs = F.log_softmax(q_logits, dim=-1)
            
            target_behavior_probs = behavior_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze()
            target_q_probs = q_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze()
            
            combined_log_prob = target_behavior_probs + self.beta * target_q_probs
            
            return combined_log_prob.mean().item()
    
    def generate_trajectory(self, task_description: str, initial_state: str, 
                           max_steps: int = 10, temperature: float = 0.7,
                           use_beam_search: bool = False) -> List[Dict[str, str]]:
        
        trajectory = []
        current_state = initial_state
        
        print(f"Task: {task_description}")
        print(f"Initial state: {initial_state[:100]}...")
        print(f"\nGenerating trajectory using Q-SFT policy extraction (β={self.beta})...")
        
        for step in range(max_steps):
            print(f"\nStep {step + 1}:")
            print(f"Current state: {current_state[:100]}...")
            
            if use_beam_search:
                action = self.generate_action_beam_search(current_state)
            else:
                action = self.generate_action(current_state, temperature=temperature)
            
            print(f"Generated action: {action}")
            
            trajectory.append({
                'step': step + 1,
                'state': current_state,
                'action': action,
                'task_description': task_description
            })
            
            current_state = f"State after action: {action[:50]}..."
            
            if any(keyword in action.lower() for keyword in ['completed', 'finished', 'done', 'found', 'success']):
                print("Task appears to be completed!")
                break
            
            if not action.strip():
                print("Warning: Empty action generated, stopping trajectory.")
                break
        
        return trajectory
    
    def compare_policies(self, state: str, top_k: int = 5) -> Dict[str, Any]:
        input_text = f"State: {state}\nAction:"
        behavior_probs, q_probs = self.get_next_token_probabilities(input_text)
        combined_probs = self.extract_policy_probabilities(input_text)
        
        behavior_top_k = torch.topk(behavior_probs, top_k)
        q_top_k = torch.topk(q_probs, top_k)
        combined_top_k = torch.topk(combined_probs, top_k)
        
        def decode_top_k(values, indices):
            return [
                {
                    'token': self.tokenizer.decode([idx.item()]),
                    'prob': val.item()
                }
                for val, idx in zip(values, indices)
            ]
        
        return {
            'behavior_policy': decode_top_k(behavior_top_k.values, behavior_top_k.indices),
            'q_value_model': decode_top_k(q_top_k.values, q_top_k.indices),
            'combined_policy': decode_top_k(combined_top_k.values, combined_top_k.indices),
            'beta': self.beta
        }
    
    def save_trajectory(self, trajectory: List[Dict[str, str]], output_path: str):
        """Save trajectory to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
        print(f"Trajectory saved to {output_path}")
    
    def set_beta(self, new_beta: float):
        """Update the beta hyperparameter for policy extraction."""
        self.beta = new_beta
        print(f"Updated β to {self.beta}")

def main():
    parser = argparse.ArgumentParser(description="Q-SFT Inference with Policy Extraction")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained Q-SFT model checkpoint")
    parser.add_argument("--task", type=str, default="Find a recipe for chocolate cake",
                       help="Task description")
    parser.add_argument("--initial_state", type=str, 
                       default="You are on a recipe website homepage with search functionality",
                       help="Initial state description")
    parser.add_argument("--max_steps", type=int, default=5,
                       help="Maximum number of steps to generate")
    parser.add_argument("--output", type=str, default="generated_trajectory.json",
                       help="Output file for generated trajectory")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Policy extraction hyperparameter β")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--use_beam_search", action="store_true",
                       help="Use beam search for generation")
    parser.add_argument("--compare_policies", action="store_true",
                       help="Compare different policies for a single state")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = QSFTInference(args.model_path, beta=args.beta)
    
    if args.compare_policies:
        print(f"\nComparing policies for state: {args.initial_state}")
        comparison = inference.compare_policies(args.initial_state)
        
        print(f"\nBehavior Policy (π_φ) top actions:")
        for i, action in enumerate(comparison['behavior_policy']):
            print(f"  {i+1}. '{action['token']}' (prob: {action['prob']:.4f})")
        
        print(f"\nQ-Value Model (p_θ) top actions:")
        for i, action in enumerate(comparison['q_value_model']):
            print(f"  {i+1}. '{action['token']}' (prob: {action['prob']:.4f})")
        
        print(f"\nCombined Policy (β={comparison['beta']}) top actions:")
        for i, action in enumerate(comparison['combined_policy']):
            print(f"  {i+1}. '{action['token']}' (prob: {action['prob']:.4f})")
    else:
        trajectory = inference.generate_trajectory(
            args.task,
            args.initial_state,
            args.max_steps,
            temperature=args.temperature,
            use_beam_search=args.use_beam_search
        )
        
        inference.save_trajectory(trajectory, args.output)
        
        print(f"\nGenerated {len(trajectory)} steps")
        print("Trajectory summary:")
        for step in trajectory:
            print(f"Step {step['step']}: {step['action'][:50]}...")

if __name__ == "__main__":
    main()