import json
import os
from datetime import datetime
from typing import Dict, Any
import sys
import copy

class TestLogger:
    """Utility class for logging test results to JSON files instead of printing to terminal"""
    
    def __init__(self, test_name: str, logs_base_dir: str = None):
        """
        Initialize the logger for a specific test
        
        Args:
            test_name: Name of the test (e.g., 'huggingface_leaderboard')
            logs_base_dir: Base directory for logs (defaults to logs/ in project root)
        """
        self.test_name = test_name
        
        # Default to logs directory in project root
        if logs_base_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logs_base_dir = os.path.join(project_root, 'logs')
        
        self.logs_base_dir = logs_base_dir
        self.test_log_dir = os.path.join(logs_base_dir, test_name)
        
        # Create directories if they don't exist
        os.makedirs(self.test_log_dir, exist_ok=True)
        
        # Generate filename with date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{test_name}_{timestamp}.json"
        self.log_filepath = os.path.join(self.test_log_dir, self.log_filename)
    
    def _remove_image_urls(self, data: Any) -> Any:
        """
        Recursively remove image_url fields from data structure while preserving reasoning steps
        
        Args:
            data: The data structure to clean
            
        Returns:
            Cleaned data structure with image_url fields removed but reasoning preserved
        """
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if key == "image_url":
                    # Skip image_url fields entirely to avoid logging large base64 data
                    continue
                else:
                    # Preserve all other fields including reasoning_steps
                    cleaned[key] = self._remove_image_urls(value)
            return cleaned
        elif isinstance(data, list):
            return [self._remove_image_urls(item) for item in data]
        else:
            return data
    
    def _count_reasoning_steps(self, data: Any) -> int:
        """
        Recursively count reasoning steps in the data structure
        
        Args:
            data: The data structure to analyze
            
        Returns:
            Total number of reasoning steps found
        """
        count = 0
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "reasoning_steps" and isinstance(value, list):
                    count += len(value)
                else:
                    count += self._count_reasoning_steps(value)
        elif isinstance(data, list):
            for item in data:
                count += self._count_reasoning_steps(item)
        return count
    
    def _count_action_steps(self, data: Any) -> int:
        """
        Recursively count action steps in the data structure
        
        Args:
            data: The data structure to analyze
            
        Returns:
            Total number of action steps found
        """
        count = 0
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "action_steps" and isinstance(value, list):
                    count += len(value)
                else:
                    count += self._count_action_steps(value)
        elif isinstance(data, list):
            for item in data:
                count += self._count_action_steps(item)
        return count
    
    def log_test_results(self, results: Dict[str, Any], additional_info: Dict[str, Any] = None):
        """
        Log test results to JSON file
        
        Args:
            results: The test results dictionary
            additional_info: Additional information to include in the log
        """
        # Create a deep copy and remove image_url fields to avoid logging large base64 data
        cleaned_results = self._remove_image_urls(copy.deepcopy(results))
        
        # Count reasoning and action steps for logging summary
        reasoning_count = 0
        action_count = 0
        if isinstance(cleaned_results, dict):
            reasoning_count = self._count_reasoning_steps(cleaned_results)
            action_count = self._count_action_steps(cleaned_results)
        
        log_entry = {
            "test_name": self.test_name,
            "timestamp": datetime.now().isoformat(),
            "results": cleaned_results
        }
        
        if additional_info:
            cleaned_additional_info = self._remove_image_urls(copy.deepcopy(additional_info))
            log_entry["additional_info"] = cleaned_additional_info
        
        try:
            with open(self.log_filepath, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
            
            print(f"Test results logged to: {self.log_filepath}")
            if reasoning_count > 0 or action_count > 0:
                step_info = []
                if reasoning_count > 0:
                    step_info.append(f"{reasoning_count} reasoning steps")
                if action_count > 0:
                    step_info.append(f"{action_count} action steps")
                print(f"Captured {' and '.join(step_info)}")
            return self.log_filepath
            
        except Exception as e:
            print(f"Error logging test results: {e}")
            # Fall back to printing results if logging fails
            self._print_results_fallback(results)
            return None
    
    def _print_results_fallback(self, results: Dict[str, Any]):
        """Fallback method to print results if JSON logging fails"""
        print(f"\n=== {self.test_name.upper()} TEST RESULTS ===")
        print(json.dumps(results, indent=2, default=str))
    
    def log_and_print_summary(self, results: Dict[str, Any], additional_info: Dict[str, Any] = None):
        """
        Log results to JSON and also print a brief summary to terminal
        
        Args:
            results: The test results dictionary
            additional_info: Additional information to include in the log
        """
        # Log to JSON file
        log_path = self.log_test_results(results, additional_info)
        
        # Print brief summary to terminal
        overall_passed = results.get('overall_passed', False)
        summary = results.get('summary', {})
        
        print(f"\n=== {self.test_name.upper()} TEST SUMMARY ===")
        print(f"Overall Result: {'PASSED' if overall_passed else 'FAILED'}")
        
        if 'passed_tests' in summary and 'total_tests' in summary:
            print(f"Tests passed: {summary['passed_tests']}/{summary['total_tests']}")
        
        if log_path:
            print(f"Detailed results: {log_path}")
        
        return log_path

def get_test_name_from_file(file_path: str = None) -> str:
    """
    Extract test name from the current file path
    
    Args:
        file_path: Path to the test file (defaults to current file)
        
    Returns:
        Test name extracted from filename
    """
    if file_path is None:
        # Get the calling file's path
        import inspect
        frame = inspect.currentframe().f_back
        file_path = frame.f_globals.get('__file__', '')
    
    filename = os.path.basename(file_path)
    # Remove 'test_' prefix and '.py' suffix
    test_name = filename.replace('test_', '').replace('.py', '')
    return test_name