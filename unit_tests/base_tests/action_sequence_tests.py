"""
Action Sequence Test Mixin
Purpose: Verify actions were called in correct order
"""
from typing import List, Any, Dict, Union, Optional


class ActionSequenceTestMixin:
    """Mixin for testing action sequences in agent trajectories."""
    
    def assertActionSequence(self, expected_sequence: List[Union[str, Any]], 
                           actual_trajectory: List[Any]) -> None:
        """
        Verify that actions were called in the correct order.
        
        Args:
            expected_sequence: List of expected actions (can be strings or action objects)
            actual_trajectory: The actual trajectory from the agent
            
        Raises:
            AssertionError: If action sequence doesn't match expected order
        """
        # Extract actions from trajectory
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and 'action' in item:
                actual_actions.append(item['action'])
            elif hasattr(item, 'action'):
                actual_actions.append(item.action)
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                # Try to convert to string representation
                actual_actions.append(str(item))
        
        # Convert expected sequence to comparable format
        expected_actions = []
        for action in expected_sequence:
            if isinstance(action, str):
                expected_actions.append(self._normalize_action_string(action))
            else:
                expected_actions.append(str(action))
        
        # Compare sequences
        self.assertEqual(len(expected_actions), len(actual_actions),
                        f"Expected {len(expected_actions)} actions, got {len(actual_actions)}")
        
        for i, (expected, actual) in enumerate(zip(expected_actions, actual_actions)):
            self.assertEqual(expected, actual, 
                           f"Action {i} mismatch: expected {expected}, got {actual}")
    
    def assertActionSequenceContains(self, expected_actions: List[Union[str, Any]], 
                                   actual_trajectory: List[Any]) -> None:
        """
        Verify that trajectory contains all expected actions in any order.
        
        Args:
            expected_actions: List of expected actions
            actual_trajectory: The actual trajectory from the agent
        """
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and 'action' in item:
                actual_actions.append(str(item['action']))
            elif hasattr(item, 'action'):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))
        
        for expected_action in expected_actions:
            if isinstance(expected_action, str):
                expected_action = self._normalize_action_string(expected_action)
            else:
                expected_action = str(expected_action)
            
            self.assertIn(expected_action, actual_actions,
                         f"Expected action {expected_action} not found in trajectory")
    
    def assertActionSequencePrefix(self, expected_prefix: List[Union[str, Any]], 
                                 actual_trajectory: List[Any]) -> None:
        """
        Verify that trajectory starts with the expected action sequence.
        
        Args:
            expected_prefix: List of expected actions at the beginning
            actual_trajectory: The actual trajectory from the agent
        """
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and 'action' in item:
                actual_actions.append(str(item['action']))
            elif hasattr(item, 'action'):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))
        
        expected_actions = []
        for action in expected_prefix:
            if isinstance(action, str):
                expected_actions.append(self._normalize_action_string(action))
            else:
                expected_actions.append(str(action))
        
        self.assertGreaterEqual(len(actual_actions), len(expected_actions),
                               f"Trajectory too short: expected at least {len(expected_actions)} actions")
        
        for i, expected in enumerate(expected_actions):
            self.assertEqual(expected, actual_actions[i],
                           f"Prefix mismatch at position {i}: expected {expected}, got {actual_actions[i]}")
    
    def assertActionSequenceSuffix(self, expected_suffix: List[Union[str, Any]], 
                                 actual_trajectory: List[Any]) -> None:
        """
        Verify that trajectory ends with the expected action sequence.
        
        Args:
            expected_suffix: List of expected actions at the end
            actual_trajectory: The actual trajectory from the agent
        """
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and 'action' in item:
                actual_actions.append(str(item['action']))
            elif hasattr(item, 'action'):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))
        
        expected_actions = []
        for action in expected_suffix:
            if isinstance(action, str):
                expected_actions.append(self._normalize_action_string(action))
            else:
                expected_actions.append(str(action))
        
        self.assertGreaterEqual(len(actual_actions), len(expected_actions),
                               f"Trajectory too short: expected at least {len(expected_actions)} actions")
        
        for i, expected in enumerate(expected_actions):
            actual_index = len(actual_actions) - len(expected_actions) + i
            self.assertEqual(expected, actual_actions[actual_index],
                           f"Suffix mismatch at position {i}: expected {expected}, got {actual_actions[actual_index]}")
    
    def assertActionFrequency(self, expected_frequency: Dict[str, int], 
                            actual_trajectory: List[Any]) -> None:
        """
        Verify that actions occur with expected frequency.
        
        Args:
            expected_frequency: Dict mapping action strings to expected counts
            actual_trajectory: The actual trajectory from the agent
        """
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and 'action' in item:
                actual_actions.append(str(item['action']))
            elif hasattr(item, 'action'):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))
        
        actual_frequency = {}
        for action in actual_actions:
            action_type = self._extract_action_type(action)
            actual_frequency[action_type] = actual_frequency.get(action_type, 0) + 1
        
        for action_type, expected_count in expected_frequency.items():
            actual_count = actual_frequency.get(action_type, 0)
            self.assertEqual(expected_count, actual_count,
                           f"Action '{action_type}' frequency mismatch: expected {expected_count}, got {actual_count}")
    
    def assertActionPattern(self, expected_pattern: str, actual_trajectory: List[Any]) -> None:
        """
        Verify that actions match a specific pattern (regex).
        
        Args:
            expected_pattern: Regular expression pattern to match
            actual_trajectory: The actual trajectory from the agent
        """
        import re
        
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and 'action' in item:
                actual_actions.append(str(item['action']))
            elif hasattr(item, 'action'):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))
        
        # Check if any action matches the pattern
        pattern_matches = []
        for action in actual_actions:
            if re.search(expected_pattern, action):
                pattern_matches.append(action)
        
        self.assertGreater(len(pattern_matches), 0,
                          f"No actions match pattern '{expected_pattern}' in trajectory")
    
    def assertActionBetween(self, action_name: str, start_action: str, end_action: str, 
                          actual_trajectory: List[Any]) -> None:
        """
        Verify that an action occurs between two other actions.
        
        Args:
            action_name: Name of the action to check
            start_action: Action that should come before
            end_action: Action that should come after
            actual_trajectory: The actual trajectory from the agent
        """
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and 'action' in item:
                actual_actions.append(str(item['action']))
            elif hasattr(item, 'action'):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))
        
        try:
            start_index = actual_actions.index(start_action)
            end_index = actual_actions.index(end_action)
            target_index = actual_actions.index(action_name)
            
            self.assertGreater(target_index, start_index,
                             f"Action '{action_name}' was called before '{start_action}'")
            self.assertLess(target_index, end_index,
                           f"Action '{action_name}' was called after '{end_action}'")
        except ValueError as e:
            self.fail(f"Could not find required actions in trajectory: {str(e)}")
    
    def assertActionNotBetween(self, action_name: str, start_action: str, end_action: str, 
                             actual_trajectory: List[Any]) -> None:
        """
        Verify that an action does NOT occur between two other actions.
        
        Args:
            action_name: Name of the action to check
            start_action: Action that should come before
            end_action: Action that should come after
            actual_trajectory: The actual trajectory from the agent
        """
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and 'action' in item:
                actual_actions.append(str(item['action']))
            elif hasattr(item, 'action'):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))
        
        try:
            start_index = actual_actions.index(start_action)
            end_index = actual_actions.index(end_action)
            
            # Check if target action appears between start and end
            for i in range(start_index + 1, end_index):
                if actual_actions[i] == action_name:
                    self.fail(f"Action '{action_name}' was called between '{start_action}' and '{end_action}'")
        except ValueError as e:
            self.fail(f"Could not find required actions in trajectory: {str(e)}")
    
    def _normalize_action_string(self, action_str: str) -> str:
        """
        Normalize action string for comparison.
        
        Args:
            action_str: Action string to normalize
            
        Returns:
            Normalized action string
        """
        # Remove extra whitespace and normalize quotes
        normalized = action_str.strip()
        normalized = normalized.replace("'", '"')
        return normalized
    
    def _extract_action_type(self, action_str: str) -> str:
        """
        Extract action type from action string.
        
        Args:
            action_str: Action string
            
        Returns:
            Action type (e.g., 'click', 'type', 'scroll')
        """
        action_lower = action_str.lower()
        
        # Common action types
        action_types = {
            'click': ['click', 'tap', 'press'],
            'type': ['type', 'input', 'send_keys', 'fill'],
            'scroll': ['scroll', 'swipe', 'move'],
            'wait': ['wait', 'sleep', 'delay', 'pause'],
            'navigate': ['navigate', 'goto', 'visit', 'open'],
            'select': ['select', 'choose', 'pick'],
            'submit': ['submit', 'send', 'post'],
            'stop': ['stop', 'end', 'finish', 'complete']
        }
        
        for action_type, keywords in action_types.items():
            if any(keyword in action_lower for keyword in keywords):
                return action_type
        
        return 'other' 