#!/usr/bin/env python3

import json
import os
import glob
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import re
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

try:
    client = OpenAI()
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")
    client = None

@dataclass
class QSFTTransition:
    state: str  # Current webpage HTML content and URL
    action: str  # Agent's action description
    reward: float  # Reward (sparse, mostly 0, 1 at completion)
    next_state: str  # Next webpage HTML content and URL
    task_description: str  # Original task for context
    instance_id: str  # Instance identifier
    step_number: int  # Step number in trajectory
    total_steps: int  # Total steps in trajectory
    task_completed: bool  # Whether task was completed

class TaskCompletionDetector:
    
    def __init__(self, use_cache: bool = True, log_unsure_instances: bool = True):
        self.use_cache = use_cache
        self.log_unsure_instances = log_unsure_instances
        self.completion_cache = {}  
        self.unsure_instances = []  
    
    def detect_completion(self, task_description: str, final_state: str, final_action: str,
                         final_url: str, instance_id: str = None) -> bool:
        """
        Detect if task was completed based on task description and final state.
        
        Args:
            task_description: Original task description
            final_state: Final webpage state
            final_action: Final action taken
            final_url: Final URL
            
        Returns:
            True if task appears to be completed
        """
        cache_key = f"{task_description[:100]}_{final_url}_{final_action[:50]}"
        
        if self.use_cache and cache_key in self.completion_cache:
            return self.completion_cache[cache_key]
        
        if client is None:
            print("Warning: OpenAI client not available, using fallback completion detection.")
            completion = self._fallback_completion_check(task_description, final_state, final_url)
            if self.use_cache:
                self.completion_cache[cache_key] = completion
            return completion
        
        try:
            class CompletionResult(BaseModel):
                completion_status: int  # 1 for completed, 0 for not completed, -1 for unsure
            
            response = client.responses.parse(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system", 
                        "content": "You are a task completion detector for web navigation tasks. You must be very strict about what constitutes completion. A task is ONLY completed if the user's specific request has been fully satisfied. Return 1 if the task is completed, 0 if not completed, or -1 if you are unsure."
                    },
                    {
                        "role": "user",
                        "content": f"""Task description: {task_description}

Final webpage state: {final_state[:2000]}...

Final action: {final_action}

Final URL: {final_url}

CRITICAL: The task is ONLY completed if the user's specific request has been fully satisfied. Consider these failure conditions:
- If the final action contains words like "error", "parsing error", "stop", "failed", "cannot", "unable" - this is a FAILURE
- If the final action indicates the agent gave up or couldn't complete the task - this is a FAILURE  
- If the final state doesn't contain the specific information requested in the task - this is a FAILURE
- If the agent stopped before finding the requested information - this is a FAILURE
- If the final URL is not relevant to the task - this is a FAILURE

The task is completed ONLY if:
- The final action was successful (no errors)
- The final state contains the specific information requested
- The user's request has been fully satisfied

Based on this strict criteria, was the task completed successfully? Return only: 1 (completed), 0 (not completed), or -1 (unsure)""",
                    },
                ],
                text_format=CompletionResult,
            )
            
            result = response.output_parsed.completion_status
            
            if result == 1:
                completion = True
            elif result == 0:
                completion = False
            else:  
                if self.log_unsure_instances:
                    unsure_instance = {
                        "task_description": task_description,
                        "final_action": final_action,
                        "final_url": final_url,
                        "final_state_preview": final_state[:500] + "..." if len(final_state) > 500 else final_state
                    }
                    if instance_id:
                        unsure_instance["instance_id"] = instance_id
                    self.unsure_instances.append(unsure_instance)
                
                completion = self._fallback_completion_check(task_description, final_state, final_url)
            
            if self.use_cache:
                self.completion_cache[cache_key] = completion
            
            return completion
            
        except Exception as e:
            print(f"Warning: LLM completion detection failed: {e}")
            return self._fallback_completion_check(task_description, final_state, final_url)
    
    def _fallback_completion_check(self, task_description: str, final_state: str, final_url: str) -> bool:
        task_lower = task_description.lower()
        state_lower = final_state.lower()
        url_lower = final_url.lower()
        
        if "recipe" in task_lower:
            recipe_indicators = ["ingredients", "directions", "prep time", "cook time", "servings"]
            return any(indicator in state_lower for indicator in recipe_indicators)
        
        elif "find" in task_lower or "locate" in task_lower:        
            search_terms = re.findall(r'"([^"]*)"', task_lower)
            if search_terms:
                return any(term.lower() in state_lower for term in search_terms)
            return "search results" in state_lower or "found" in state_lower
        
        elif "purchase" in task_lower or "buy" in task_lower:
            shopping_indicators = ["cart", "checkout", "purchase", "order"]
            return any(indicator in state_lower for indicator in shopping_indicators)
        
        else:
            task_words = [word for word in task_lower.split() if len(word) > 3]
            return any(word in state_lower for word in task_words)
    
    def save_unsure_instances(self, output_path: str = "unsure_instances.json"):
        if self.unsure_instances:
            with open(output_path, 'w') as f:
                json.dump(self.unsure_instances, f, indent=2)
            print(f"Saved {len(self.unsure_instances)} unsure instances to {output_path}")
        else:
            print("No unsure instances to save.")

        
class WebArenaToQSFTMapper:
    def __init__(self, instances_dir: str):
        self.instances_dir = Path(instances_dir)
        self.completion_detector = TaskCompletionDetector()
        
    def load_instance_metadata(self, instance_id: str) -> Dict:
        metadata_path = self.instances_dir / instance_id / "metadata.json"
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_agent_points(self, instance_id: str) -> List[Dict]:
        points_path = self.instances_dir / instance_id / "agent" / "points.json"
        with open(points_path, 'r') as f:
            return json.load(f)
    
    def load_observation_data(self, instance_id: str, timestamp: str) -> Dict:
        obs_file = f"{instance_id}_agent_observation_{timestamp}.json"
        obs_path = self.instances_dir / instance_id / "agent" / "json_data" / obs_file
        
        with open(obs_path, 'r') as f:
            return json.load(f)
    
    def load_action_data(self, instance_id: str, timestamp: str) -> Dict:
        action_file = f"{instance_id}_agent_action_{timestamp}.json"
        action_path = self.instances_dir / instance_id / "agent" / "json_data" / action_file
        
        with open(action_path, 'r') as f:
            return json.load(f)
    
    def load_user_action_data(self, instance_id: str) -> str:
        user_data_dir = self.instances_dir / instance_id / "user" / "json_data"
        
        user_action_files = list(user_data_dir.glob(f"{instance_id}_user_action_*.json"))
        
        if not user_action_files:
            print(f"Warning: No user action file found for {instance_id}")
            return ""
        
        user_action_file = user_action_files[0]
        
        with open(user_action_file, 'r') as f:
            user_data = json.load(f)
        
        task_description = user_data.get("text", "")
        return task_description
    
    def extract_state_representation(self, observation_data: Dict) -> str:
        url = observation_data.get("url", "")
        html = observation_data.get("html", "")
        
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        title = title_match.group(1) if title_match else ""
        
        visible_text = self.extract_visible_text(html)
        
        interactive_elements = self.extract_interactive_elements(html)
                
        state_parts = [f"URL: {url}"]
        
        if title:
            state_parts.append(f"Title: {title}")
        
        state_parts.append(f"\nPage Content:\n{visible_text}")
        
        if interactive_elements:
            state_parts.append(f"\nInteractive Elements:\n{interactive_elements}")
        
        return "\n".join(state_parts)
    
    def extract_visible_text(self, html: str) -> str:
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\[[^\]]+\]', '', text)  
        text = re.sub(r'\s+', ' ', text)  
        text = text.strip()
        
        if len(text) > 3000:
            text = text[:3000] + "..."
        
        return text
    
    def extract_interactive_elements(self, html: str) -> str:
        elements = []
        
        button_pattern = r'<button[^>]*>([^<]+)</button>'
        link_pattern = r'<a[^>]*>([^<]+)</a>'
        input_pattern = r'<input[^>]*>'
        
        buttons = re.findall(button_pattern, html, re.IGNORECASE)
        links = re.findall(link_pattern, html, re.IGNORECASE)
        inputs = re.findall(input_pattern, html, re.IGNORECASE)
        
        if buttons:
            elements.append(f"Buttons: {', '.join(buttons[:5])}")
        if links:
            elements.append(f"Links: {', '.join(links[:5])}")
        if inputs:
            elements.append(f"Input fields: {len(inputs)} found")
        
        return "\n".join(elements)
    
    def extract_action_representation(self, action_data: Dict) -> str:
        
        function_type = action_data.get("function", "")
        kwargs = action_data.get("kwargs", {})
        description = action_data.get("description", "")
        
        action_parts = [f"Action: {function_type}"]
        
        if kwargs:
            kwargs_str = json.dumps(kwargs, indent=2)
            action_parts.append(f"Parameters: {kwargs_str}")
        
        if description:
            reasoning = self.extract_reasoning(description)
            if reasoning:
                action_parts.append(f"Reasoning: {reasoning}")
        
        return "\n".join(action_parts)
    
    def extract_reasoning(self, description: str) -> str:
        if "Let's think step-by-step" in description:
            lines = description.split('\n')
            reasoning_lines = []
            in_reasoning = False
            
            for line in lines:
                if "Let's think step-by-step" in line:
                    in_reasoning = True
                    continue
                elif "In summary" in line or "the next action" in line:
                    break
                elif in_reasoning:
                    reasoning_lines.append(line.strip())
            
            return " ".join(reasoning_lines)
        
        return description

    
    def map_instance_to_qsft(self, instance_id: str) -> List[QSFTTransition]:
        task_description = self.load_user_action_data(instance_id)
        
        if not task_description:
            print(f"Warning: No task description found for {instance_id}")
            return []
        
        points = self.load_agent_points(instance_id)
        
        agent_points = [p for p in points if p.get("agent_id") == "agent"]
        
        transitions = []
        step_counter = 0  
        
        final_state = ""
        final_url = ""
        final_action = ""
        if agent_points:
            last_obs = [p for p in agent_points if p["point_type"] == "observation"][-1]
            try:
                final_obs_data = self.load_observation_data(instance_id, last_obs["timestamp"])
                final_state = self.extract_state_representation(final_obs_data)
                final_url = final_obs_data.get("url", "")
            except FileNotFoundError:
                pass
            
            last_action = [p for p in agent_points if p["point_type"] == "action"][-1]
            try:
                final_action_data = self.load_action_data(instance_id, last_action["timestamp"])
                final_action = self.extract_action_representation(final_action_data)
            except FileNotFoundError:
                pass
        
        print(f"Processing {instance_id}: {len(agent_points)} agent points")
        for i, point in enumerate(agent_points):
            timestamp = point["timestamp"]
            point_type = point["point_type"]
            
            if point_type == "observation":
                try:
                    obs_data = self.load_observation_data(instance_id, timestamp)
                    state = self.extract_state_representation(obs_data)
                    
                    if i + 1 < len(agent_points) and agent_points[i + 1]["point_type"] == "action":
                        action_point = agent_points[i + 1]
                        action_timestamp = action_point["timestamp"]
                        
                        try:
                            action_data = self.load_action_data(instance_id, action_timestamp)
                            action = self.extract_action_representation(action_data)
                            
                            is_final_action = (i + 2 >= len(agent_points))
                            
                            if is_final_action:
                                next_state = final_state
                            else:
                                if i + 2 < len(agent_points) and agent_points[i + 2]["point_type"] == "observation":
                                    next_obs_point = agent_points[i + 2]
                                    next_obs_timestamp = next_obs_point["timestamp"]
                                    
                                    try:
                                        next_obs_data = self.load_observation_data(instance_id, next_obs_timestamp)
                                        next_state = self.extract_state_representation(next_obs_data)
                                    except FileNotFoundError:
                                        print(f"Warning: Could not load next observation for {instance_id} at {next_obs_timestamp}")
                                        continue
                                else:
                                    continue
                            
                            reward = 0.0
                            task_completed = False
                            
                            transition = QSFTTransition(
                                state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                task_description=task_description,
                                instance_id=instance_id,
                                step_number=step_counter,
                                total_steps=len(agent_points),
                                task_completed=task_completed
                            )
                            
                            transitions.append(transition)
                            step_counter += 1  
                            print(f"  Created transition {step_counter-1} at index {i}")
                                    
                        except FileNotFoundError:
                            print(f"Warning: Could not load action for {instance_id} at {action_timestamp}")
                            continue
                            
                except FileNotFoundError:
                    print(f"Warning: Could not load observation for {instance_id} at {timestamp}")
                    continue
        
        if transitions:
            task_completed = self.completion_detector.detect_completion(
                task_description, final_state, final_action, final_url, instance_id
            )
            final_reward = 1.0 if task_completed else 0.0
            
            transitions[-1].reward = final_reward
            transitions[-1].task_completed = task_completed
        
        return transitions
    
    def map_all_instances(self) -> List[QSFTTransition]:
        all_transitions = []
        
        instance_dirs = [d for d in self.instances_dir.iterdir() if d.is_dir()]
        
        print(f"Processing {len(instance_dirs)} instances...")
        
        for instance_dir in instance_dirs:
            instance_id = instance_dir.name
            
            try:
                transitions = self.map_instance_to_qsft(instance_id)
                all_transitions.extend(transitions)
                print(f"Processed {instance_id}: {len(transitions)} transitions")
                
            except Exception as e:
                print(f"Error processing {instance_id}: {e}")
                continue
        
        print(f"Total transitions: {len(all_transitions)}")
        
        if hasattr(self.completion_detector, 'unsure_instances') and self.completion_detector.unsure_instances:
            self.completion_detector.save_unsure_instances()
        
        return all_transitions
    
    def save_qsft_dataset(self, transitions: List[QSFTTransition], output_path: str):
        dataset = []
        
        for transition in transitions:
            dataset.append({
                "state": transition.state,
                "action": transition.action,
                "reward": transition.reward,
                "next_state": transition.next_state,
                "task_description": transition.task_description,
                "instance_id": transition.instance_id,
                "step_number": transition.step_number,
                "total_steps": transition.total_steps,
                "task_completed": transition.task_completed
            })
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved Q-SFT dataset to {output_path}")
        print(f"Dataset contains {len(dataset)} transitions")
    
    def generate_statistics(self, transitions: List[QSFTTransition]) -> Dict[str, Any]:
        """Generate statistics about the dataset."""
        if not transitions:
            return {}
        
        total_transitions = len(transitions)
        completed_tasks = sum(1 for t in transitions if t.task_completed)
        total_reward = sum(t.reward for t in transitions)
        avg_reward = total_reward / total_transitions if total_transitions > 0 else 0
        
        unique_instances = set(t.instance_id for t in transitions)
        completed_instances = set(t.instance_id for t in transitions if t.task_completed)
        completion_rate = len(completed_instances) / len(unique_instances) if unique_instances else 0
        
        instance_lengths = {}
        for t in transitions:
            if t.instance_id not in instance_lengths:
                instance_lengths[t.instance_id] = t.total_steps
        
        avg_trajectory_length = sum(instance_lengths.values()) / len(instance_lengths) if instance_lengths else 0
        
        return {
            "total_transitions": total_transitions,
            "completed_tasks": completed_tasks,
            "completion_rate": completion_rate,
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "average_trajectory_length": avg_trajectory_length,
            "unique_instances": len(unique_instances),
            "completed_instances": len(completed_instances)
        }

def main():
    instances_dir = ".data/webarena/instances"
    output_path = "webarena_qsft_dataset.json"
    
    mapper = WebArenaToQSFTMapper(instances_dir)
    
    transitions = mapper.map_all_instances()
    
    mapper.save_qsft_dataset(transitions, output_path)
    
    stats = mapper.generate_statistics(transitions)
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    if transitions:
        print("\nSample transitions:")
        for i, transition in enumerate(transitions[:3]):
            print(f"\nTransition {i+1}:")
            print(f"Task: {transition.task_description}")
            print(f"Step: {transition.step_number}/{transition.total_steps}")
            print(f"State length: {len(transition.state)} chars")
            print(f"Action: {transition.action[:100]}...")
            print(f"Reward: {transition.reward}")
            print(f"Task completed: {transition.task_completed}")

if __name__ == "__main__":
    main() 