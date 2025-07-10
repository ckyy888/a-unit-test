import argparse
import os
import sys
from pathlib import Path
import logging
import json
import torch
import time
from typing import List, Dict, Any, Optional
import traceback

try:
    from qsft_trainer import QSFTTrainer, QSFTTransition
except ImportError as e:
    print(f"Error: Could not import QSFTTrainer: {e}")
    print("Make sure qsft_trainer.py is in the same directory or Python path.")
    sys.exit(1)

try:
    from webvoyager_mapper import WebVoyagerToQSFTMapper
    WEBVOYAGER_AVAILABLE = True
except ImportError:
    print("Warning: webvoyager_mapper not found. Dataset generation will be skipped.")
    WebVoyagerToQSFTMapper = None
    WEBVOYAGER_AVAILABLE = False

def setup_logging(log_level=logging.INFO, log_file=None):
    formatters = {
        'detailed': logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ),
        'simple': logging.Formatter('%(levelname)s: %(message)s')
    }
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatters['simple'])
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatters['detailed'])
        logger.addHandler(file_handler)
    
    return logger

logger = logging.getLogger(__name__)

def validate_arguments(args):
    errors = []
    
    if not args.skip_dataset_generation and not args.eval_only:
        if not WEBVOYAGER_AVAILABLE:
            errors.append("WebVoyager mapper not available but dataset generation requested")
        elif not os.path.exists(args.instances_dir):
            errors.append(f"Instances directory not found: {args.instances_dir}")
    
    if args.gamma < 0 or args.gamma > 1:
        errors.append(f"Gamma must be between 0 and 1, got {args.gamma}")
    
    if args.beta < 0:
        errors.append(f"Beta must be non-negative, got {args.beta}")
    
    if args.target_update_rate <= 0 or args.target_update_rate > 1:
        errors.append(f"Target update rate must be in (0, 1], got {args.target_update_rate}")
    
    if args.eval_split < 0 or args.eval_split >= 1:
        errors.append(f"Eval split must be in [0, 1), got {args.eval_split}")
    
    if args.batch_size <= 0:
        errors.append(f"Batch size must be positive, got {args.batch_size}")
    
    if args.learning_rate <= 0:
        errors.append(f"Learning rate must be positive, got {args.learning_rate}")

    if args.eval_only and not args.eval_checkpoint:
        checkpoint_path = os.path.join(args.model_output_dir, "checkpoint-final")
        if not os.path.exists(checkpoint_path):
            errors.append(f"Evaluation requested but no checkpoint found at {checkpoint_path}")
    
    if errors:
        logger.error("Argument validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="Complete Q-SFT Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    dataset_group = parser.add_argument_group('Dataset Generation')
    dataset_group.add_argument("--instances_dir", type=str, 
                              default=".data/webvoyager-nnetnav/instances",
                              help="Path to WebVoyager instances directory")
    dataset_group.add_argument("--dataset_output", type=str, 
                              default="webvoyager_qsft_dataset.json",
                              help="Output path for Q-SFT dataset")
    
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--model_name", type=str, default="gpt2-medium",
                            help="Pretrained model name to use")
    model_group.add_argument("--model_output_dir", type=str, default="qsft_model",
                            help="Directory to save the trained model")
    
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument("--num_epochs", type=int, default=3,
                               help="Number of training epochs for Q-value model")
    training_group.add_argument("--behavior_epochs", type=int, default=2,
                               help="Number of training epochs for behavior policy")
    training_group.add_argument("--batch_size", type=int, default=4,
                               help="Training batch size")
    training_group.add_argument("--learning_rate", type=float, default=5e-5,
                               help="Learning rate")
    training_group.add_argument("--behavior_lr", type=float, default=None,
                               help="Learning rate for behavior policy (uses --learning_rate if not set)")
    training_group.add_argument("--max_length", type=int, default=512,
                               help="Maximum sequence length")
    training_group.add_argument("--save_steps", type=int, default=500,
                               help="Save checkpoint every N steps")
    training_group.add_argument("--eval_split", type=float, default=0.1,
                               help="Fraction of data to use for evaluation")
    training_group.add_argument("--warmup_steps", type=int, default=100,
                               help="Number of warmup steps for learning rate scheduler")
    
    qsft_group = parser.add_argument_group('Q-SFT Hyperparameters')
    qsft_group.add_argument("--gamma", type=float, default=0.99,
                           help="Discount factor for Bellman equation")
    qsft_group.add_argument("--beta", type=float, default=1.0,
                           help="Policy extraction hyperparameter")
    qsft_group.add_argument("--target_update_rate", type=float, default=0.005,
                           help="Target network update rate (α)")
    
    control_group = parser.add_argument_group('Execution Control')
    control_group.add_argument("--skip_dataset_generation", action="store_true",
                              help="Skip dataset generation if dataset already exists")
    control_group.add_argument("--skip_training", action="store_true",
                              help="Skip training if only want to generate dataset")
    control_group.add_argument("--skip_behavior_training", action="store_true",
                              help="Skip behavior policy training (use pretrained weights)")
    control_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                              help="Resume training from a specific checkpoint")
    control_group.add_argument("--force_overwrite", action="store_true",
                              help="Overwrite existing outputs without prompting")
    
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument("--eval_only", action="store_true",
                           help="Only run evaluation on existing model")
    eval_group.add_argument("--eval_checkpoint", type=str, default=None,
                           help="Specific checkpoint to evaluate")
    eval_group.add_argument("--eval_tasks", type=str, nargs='+', 
                           default=["Find a chocolate cake recipe", "Navigate to product page"],
                           help="Tasks to evaluate the model on")
    eval_group.add_argument("--eval_beta_values", type=float, nargs='+',
                           default=None, help="Beta values to test during evaluation")
    
    debug_group = parser.add_argument_group('Logging and Debugging')
    debug_group.add_argument("--log_file", type=str, default=None,
                            help="File to save detailed logs")
    debug_group.add_argument("--log_level", type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            help="Logging level")
    debug_group.add_argument("--debug", action="store_true",
                            help="Enable debug mode with additional logging")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    if args.debug:
        log_level = logging.DEBUG
    
    log_file = args.log_file
    if log_file is None and args.model_output_dir:
        log_file = os.path.join(args.model_output_dir, "pipeline.log")
    
    setup_logging(log_level, log_file)
    
    logger.info("=" * 60)
    logger.info("Q-SFT Training Pipeline Starting")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    
    if args.behavior_lr is None:
        args.behavior_lr = args.learning_rate

    if not validate_arguments(args):
        logger.error("Argument validation failed. Exiting.")
        return 1
    
    try:
        os.makedirs(args.model_output_dir, exist_ok=True)
        
        config_path = os.path.join(args.model_output_dir, "pipeline_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        if not args.skip_dataset_generation and not args.eval_only:
            success = generate_dataset_step(args)
            if not success:
                return 1
        else:
            logger.info("Skipping dataset generation.")
        
        if args.eval_only:
            logger.info("Running evaluation only...")
            success = evaluate_model_step(args)
            if not success:
                return 1
        elif not args.skip_training:
            logger.info("Step 2: Training Q-SFT models...")
            success = train_model_step(args)
            if not success:
                return 1
            
            logger.info("Step 3: Evaluating trained model...")
            success = evaluate_model_step(args)
            if not success:
                logger.warning("Evaluation failed, but training completed successfully")
        else:
            logger.info("Skipping training.")
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info("=" * 60)
        logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds!")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

def generate_dataset_step(args) -> bool:
    logger.info("Step 1: Generating Q-SFT dataset...")
    
    if not WEBVOYAGER_AVAILABLE:
        logger.error("WebVoyagerToQSFTMapper not available. Cannot generate dataset.")
        return False
    
    if os.path.exists(args.dataset_output) and not args.force_overwrite:
        logger.info(f"Dataset already exists at {args.dataset_output}")
        try:
            if not sys.stdin.isatty():  
                logger.info("Non-interactive mode: using existing dataset")
                return True
            response = input("Regenerate dataset? (y/n): ").lower()
            if response != 'y':
                logger.info("Using existing dataset.")
                return True
        except (EOFError, KeyboardInterrupt):
            logger.info("Using existing dataset.")
            return True
    
    try:
        success = generate_dataset(args.instances_dir, args.dataset_output)
        if success:
            logger.info(f"Dataset generation completed: {args.dataset_output}")
        return success
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def train_model_step(args) -> bool:
    dataset_path = args.dataset_output
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        logger.error("Please generate the dataset first or provide the correct path.")
        return False
    
    try:
        success = train_model(dataset_path, args)
        if success:
            logger.info("Model training completed successfully")
        return success
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def evaluate_model_step(args) -> bool:
    try:
        success = evaluate_model(args)
        if success:
            logger.info("Model evaluation completed successfully")
        return success
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def generate_dataset(instances_dir: str, output_path: str) -> bool:
    logger.info(f"Loading instances from {instances_dir}")
    
    if not os.path.exists(instances_dir):
        logger.error(f"Instances directory not found: {instances_dir}")
        return False
    
    try:
        mapper = WebVoyagerToQSFTMapper(instances_dir)
        
        logger.info("Converting WebVoyager instances to Q-SFT format...")
        transitions = mapper.map_all_instances()
        
        if not transitions:
            logger.error("No transitions generated. Check the instances directory and data format.")
            return False
        
        logger.info(f"Generated {len(transitions)} transitions")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mapper.save_qsft_dataset(transitions, output_path)
        
        stats = mapper.generate_statistics(transitions)
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
        
        stats_path = output_path.replace('.json', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        return False

def train_model(dataset_path: str, args) -> bool:
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        trainer = QSFTTrainer(
            model_name=args.model_name,
            gamma=args.gamma,
            beta=args.beta,
            target_update_rate=args.target_update_rate
        )
        
        transitions = trainer.load_dataset(dataset_path)
        
        if not transitions:
            logger.error("No transitions loaded from dataset.")
            return False
        
        split_idx = int(len(transitions) * (1 - args.eval_split))
        train_transitions = transitions[:split_idx]
        eval_transitions = transitions[split_idx:]
        
        logger.info(f"Train set: {len(train_transitions)} transitions")
        logger.info(f"Eval set: {len(eval_transitions)} transitions")
        
        if len(train_transitions) == 0:
            logger.error("No training transitions available")
            return False
        
        train_dataloader = trainer.create_dataloader(
            train_transitions, args.batch_size, args.max_length
        )
        eval_dataloader = trainer.create_dataloader(
            eval_transitions, args.batch_size, args.max_length
        ) if eval_transitions else None
        
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            if not load_checkpoint(trainer, args.resume_from_checkpoint):
                logger.error("Failed to load checkpoint")
                return False
        
        if not args.skip_behavior_training:
            logger.info("Phase 1: Training behavior policy π_φ...")
            trainer.train_behavior_policy(
                train_dataloader,
                num_epochs=args.behavior_epochs,
                learning_rate=args.behavior_lr
            )

            behavior_checkpoint_dir = os.path.join(args.model_output_dir, "behavior_only")
            os.makedirs(behavior_checkpoint_dir, exist_ok=True)
            trainer.behavior_model.save_pretrained(behavior_checkpoint_dir)
            trainer.tokenizer.save_pretrained(behavior_checkpoint_dir)
            logger.info(f"Behavior policy saved to {behavior_checkpoint_dir}")
        else:
            logger.info("Skipping behavior policy training (using pretrained weights)")
        
        logger.info("Phase 2: Training Q-value model p_θ...")
        trainer.train_q_values(
            train_dataloader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            output_dir=args.model_output_dir
        )
        
        if eval_dataloader:
            logger.info("Evaluating trained models on validation set...")
            eval_metrics = trainer.evaluate(eval_dataloader)
            
            logger.info("Training Evaluation Results:")
            for metric, value in eval_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            eval_results_path = os.path.join(args.model_output_dir, "training_eval_results.json")
            with open(eval_results_path, 'w') as f:
                json.dump(eval_metrics, f, indent=2)
            
            logger.info(f"Training evaluation results saved to {eval_results_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

def load_checkpoint(trainer: QSFTTrainer, checkpoint_path: str) -> bool:
    """Load models from a checkpoint with efficient reconstruction."""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
        return False
    
    try:
        success_count = 0
        
        # Load configuration first
        config_path = os.path.join(checkpoint_path, "qsft_config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                trainer.gamma = config.get('gamma', trainer.gamma)
                trainer.beta = config.get('beta', trainer.beta)
                trainer.target_update_rate = config.get('target_update_rate', trainer.target_update_rate)
            logger.info("Loaded training configuration from checkpoint")
            success_count += 1
        
        # Load behavior model (check for reference first)
        behavior_ref_path = os.path.join(checkpoint_path, "behavior_model_path.txt")
        if os.path.exists(behavior_ref_path):
            # Load from reference path
            with open(behavior_ref_path, 'r') as f:
                behavior_path = f.read().strip()
            if os.path.exists(behavior_path):
                from transformers import GPT2LMHeadModel
                trainer.behavior_model = GPT2LMHeadModel.from_pretrained(behavior_path)
                trainer.behavior_model.to(trainer.device)
                logger.info(f"Loaded behavior model from reference: {behavior_path}")
                success_count += 1
        else:
            # Fallback: load from checkpoint directly
            behavior_path = os.path.join(checkpoint_path, "behavior_model")
            if os.path.exists(behavior_path):
                from transformers import GPT2LMHeadModel
                trainer.behavior_model = GPT2LMHeadModel.from_pretrained(behavior_path)
                trainer.behavior_model.to(trainer.device)
                logger.info("Loaded behavior model from checkpoint")
                success_count += 1
        
        # Load Q-value model
        q_model_path = os.path.join(checkpoint_path, "q_model")
        if os.path.exists(q_model_path):
            from transformers import GPT2LMHeadModel
            trainer.q_model = GPT2LMHeadModel.from_pretrained(q_model_path)
            trainer.q_model.to(trainer.device)
            logger.info("Loaded Q-value model from checkpoint")
            success_count += 1
            
            # 🔄 RECONSTRUCTION LOGIC HERE
            if config.get('reconstruct_target', False):
                # Reconstruct target model from Q-value model
                import copy
                trainer.target_q_model = copy.deepcopy(trainer.q_model)
                trainer.target_q_model.to(trainer.device)
                logger.info("Reconstructed target Q-value model from Q-value model")
                success_count += 1
            else:
                # Try to load target model directly (old checkpoint format)
                target_path = os.path.join(checkpoint_path, "target_q_model")
                if os.path.exists(target_path):
                    trainer.target_q_model = GPT2LMHeadModel.from_pretrained(target_path)
                    trainer.target_q_model.to(trainer.device)
                    logger.info("Loaded target Q-value model from checkpoint")
                    success_count += 1
                else:
                    # Fallback: reconstruct anyway
                    import copy
                    trainer.target_q_model = copy.deepcopy(trainer.q_model)
                    trainer.target_q_model.to(trainer.device)
                    logger.info("Target model not found, reconstructed from Q-value model")
                    success_count += 1
        
        if success_count == 0:
            logger.error("No components successfully loaded from checkpoint")
            return False
        
        logger.info(f"Successfully loaded {success_count} components from checkpoint")
        return True
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return False

def evaluate_model(args) -> bool:
    try:
        from qsft_inference import QSFTInference
    except ImportError as e:
        logger.error(f"Could not import QSFTInference: {e}")
        return False
    
    if args.eval_checkpoint:
        checkpoint_path = args.eval_checkpoint
    else:
        checkpoint_path = os.path.join(args.model_output_dir, "checkpoint-final", "q_model")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        checkpoint_dir = args.model_output_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                logger.info(f"Using latest checkpoint: {checkpoint_path}")
            else:
                logger.error("No checkpoints found")
                return False
        else:
            return False
    
    logger.info(f"Evaluating model from {checkpoint_path}")
    
    try:
        inference = QSFTInference(checkpoint_path, beta=args.beta)
        
        model_info = inference.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        if args.eval_beta_values:
            beta_values = args.eval_beta_values
        else:
            beta_values = [0.5, 1.0, 2.0, 4.0] if args.beta == 1.0 else [args.beta]
        
        all_results = {}
        
        for beta in beta_values:
            logger.info(f"\nEvaluating with β = {beta}")
            inference.set_beta(beta)
            beta_results = {}
            
            for i, task in enumerate(args.eval_tasks):
                logger.info(f"\nTask {i+1}: {task}")
                
                try:
                    initial_state = "You are on a website homepage with navigation and search functionality"
                    trajectory = inference.generate_trajectory(
                        task_description=task,
                        initial_state=initial_state,
                        max_steps=5,
                        temperature=0.7
                    )
                    
                    output_file = f"eval_trajectory_beta_{beta}_task_{i+1}.json"
                    output_path = os.path.join(args.model_output_dir, output_file)
                    inference.save_trajectory(trajectory, output_path)
                    
                    analysis = analyze_trajectory(trajectory, task)
                    beta_results[f"task_{i+1}"] = {
                        "task": task,
                        "trajectory": trajectory,
                        "analysis": analysis,
                        "trajectory_file": output_path
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating task {i+1}: {e}")
                    beta_results[f"task_{i+1}"] = {"error": str(e)}
            
            all_results[f"beta_{beta}"] = beta_results
        
        logger.info(f"\nPolicy comparison for sample state:")
        sample_state = "You are looking at a search results page with multiple recipe options"
        comparison = inference.compare_policies(sample_state, top_k=3)
        
        if 'error' not in comparison:
            logger.info(f"\nBehavior Policy (π_φ) top actions:")
            for i, action in enumerate(comparison['behavior_policy']):
                logger.info(f"  {i+1}. '{action['token']}' (prob: {action['prob']:.4f})")
            
            logger.info(f"\nQ-Value Model (p_θ) top actions:")
            for i, action in enumerate(comparison['q_value_model']):
                logger.info(f"  {i+1}. '{action['token']}' (prob: {action['prob']:.4f})")
            
            logger.info(f"\nCombined Policy (β={comparison['beta']}) top actions:")
            for i, action in enumerate(comparison['combined_policy']):
                logger.info(f"  {i+1}. '{action['token']}' (prob: {action['prob']:.4f})")
        
        eval_summary = {
            "checkpoint_path": checkpoint_path,
            "model_info": model_info,
            "evaluation_results": all_results,
            "policy_comparison": comparison,
            "evaluation_args": {
                "beta_values": beta_values,
                "eval_tasks": args.eval_tasks
            }
        }
        
        results_path = os.path.join(args.model_output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(eval_summary, f, indent=2, default=str)
        logger.info(f"Comprehensive evaluation results saved to {results_path}")
        
        comparison_path = os.path.join(args.model_output_dir, "policy_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Policy comparison saved to {comparison_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return False

def analyze_trajectory(trajectory: List[dict], task: str) -> Dict[str, Any]:    
    if not trajectory:
        return {
            'steps': 0,
            'completed': False,
            'action_diversity': 0,
            'empty_actions': 0,
            'error': 'Empty trajectory'
        }
    
    logger.info(f"Analyzing trajectory for task: {task}")
    logger.info(f"  Steps: {len(trajectory)}")
    
    completion_keywords = ['completed', 'finished', 'done', 'found', 'success', 'recipe', 'product', 'navigate']
    
    completed = False
    completion_step = None
    for step in trajectory:
        action = step.get('action', '').lower()
        if any(keyword in action for keyword in completion_keywords):
            completed = True
            completion_step = step.get('step', 'unknown')
            break
    
    logger.info(f"  Task completion detected: {completed}")
    if completed:
        logger.info(f"  Completion detected at step: {completion_step}")
    
    actions = [step.get('action', '') for step in trajectory]
    unique_actions = len(set(actions))
    diversity_ratio = unique_actions / len(actions) if actions else 0
    logger.info(f"  Action diversity: {unique_actions}/{len(actions)} unique actions ({diversity_ratio:.2f})")
    
    empty_actions = sum(1 for action in actions if not action.strip())
    if empty_actions > 0:
        logger.warning(f"  Empty actions: {empty_actions}/{len(actions)}")

    avg_action_length = sum(len(action) for action in actions) / len(actions) if actions else 0
    
    return {
        'steps': len(trajectory),
        'completed': completed,
        'completion_step': completion_step,
        'action_diversity': diversity_ratio,
        'unique_actions': unique_actions,
        'total_actions': len(actions),
        'empty_actions': empty_actions,
        'avg_action_length': avg_action_length,
        'task': task
    }

if __name__ == "__main__":
    exit(main())