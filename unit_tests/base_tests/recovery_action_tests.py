"""
Recovery Action Test Mixin
Purpose: Verify recovery behavior was invoked
"""
from typing import Any, Callable, Dict, List, Optional, Union


class RecoveryActionTestMixin:
    """Mixin for testing recovery actions in agent trajectories."""

    def assertRecoveryAction(
        self,
        error_condition: str,
        expected_action: Union[str, Any],
        actual_action: Union[str, Any],
    ) -> None:
        """
        Verify that recovery behavior was invoked when an error condition occurred.

        Args:
            error_condition: Description of the error condition that should trigger recovery
            expected_action: The expected recovery action
            actual_action: The actual action taken by the agent

        Raises:
            AssertionError: If recovery action doesn't match expected behavior
        """
        # Normalize actions for comparison
        if isinstance(expected_action, str):
            expected_action = self._normalize_action_string(expected_action)
        if isinstance(actual_action, str):
            actual_action = self._normalize_action_string(actual_action)
        else:
            actual_action = str(actual_action)

        self.assertEqual(
            expected_action,
            actual_action,
            f"Recovery action mismatch for error '{error_condition}': "
            f"expected {expected_action}, got {actual_action}",
        )

    def assertRecoveryActionInTrajectory(
        self,
        error_condition: str,
        expected_recovery_actions: List[Union[str, Any]],
        actual_trajectory: List[Any],
    ) -> None:
        """
        Verify that recovery actions are present in trajectory after error condition.

        Args:
            error_condition: Description of the error condition
            expected_recovery_actions: List of expected recovery actions
            actual_trajectory: The actual trajectory from the agent
        """
        # Extract actions from trajectory
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and "action" in item:
                actual_actions.append(str(item["action"]))
            elif hasattr(item, "action"):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))

        # Check if any of the expected recovery actions are present
        expected_actions = []
        for action in expected_recovery_actions:
            if isinstance(action, str):
                expected_actions.append(self._normalize_action_string(action))
            else:
                expected_actions.append(str(action))

        found_recovery_actions = []
        for expected_action in expected_actions:
            if expected_action in actual_actions:
                found_recovery_actions.append(expected_action)

        self.assertGreater(
            len(found_recovery_actions),
            0,
            f"No recovery actions found for error '{error_condition}'. "
            f"Expected one of: {expected_recovery_actions}",
        )

    def assertRecoverySequence(
        self,
        error_condition: str,
        expected_recovery_sequence: List[Union[str, Any]],
        actual_trajectory: List[Any],
    ) -> None:
        """
        Verify that a specific sequence of recovery actions was executed.

        Args:
            error_condition: Description of the error condition
            expected_recovery_sequence: Expected sequence of recovery actions
            actual_trajectory: The actual trajectory from the agent
        """
        # Extract actions from trajectory
        actual_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and "action" in item:
                actual_actions.append(str(item["action"]))
            elif hasattr(item, "action"):
                actual_actions.append(str(item.action))
            elif isinstance(item, str):
                actual_actions.append(item)
            else:
                actual_actions.append(str(item))

        # Convert expected sequence to comparable format
        expected_actions = []
        for action in expected_recovery_sequence:
            if isinstance(action, str):
                expected_actions.append(self._normalize_action_string(action))
            else:
                expected_actions.append(str(action))

        # Find the recovery sequence in the actual trajectory
        sequence_found = self._find_sequence_in_trajectory(
            expected_actions, actual_actions
        )
        self.assertTrue(
            sequence_found,
            f"Recovery sequence not found for error '{error_condition}'. "
            f"Expected: {expected_recovery_sequence}",
        )

    def assertRecoveryTimeout(
        self,
        error_condition: str,
        max_recovery_time: float,
        recovery_function: Callable,
    ) -> None:
        """
        Verify that recovery actions complete within expected time limit.

        Args:
            error_condition: Description of the error condition
            max_recovery_time: Maximum allowed time for recovery in seconds
            recovery_function: Function that performs recovery actions
        """
        import time

        start_time = time.time()
        try:
            recovery_function()
            recovery_time = time.time() - start_time
            self.assertLessEqual(
                recovery_time,
                max_recovery_time,
                f"Recovery for error '{error_condition}' took {recovery_time:.2f}s, "
                f"exceeded limit of {max_recovery_time}s",
            )
        except Exception as e:
            self.fail(
                f"Recovery function failed for error '{error_condition}': {str(e)}"
            )

    def assertRecoveryRetryCount(
        self,
        error_condition: str,
        expected_retry_count: int,
        actual_retry_count: int,
    ) -> None:
        """
        Verify that recovery was attempted the expected number of times.

        Args:
            error_condition: Description of the error condition
            expected_retry_count: Expected number of retry attempts
            actual_retry_count: Actual number of retry attempts
        """
        self.assertEqual(
            expected_retry_count,
            actual_retry_count,
            f"Retry count mismatch for error '{error_condition}': "
            f"expected {expected_retry_count}, got {actual_retry_count}",
        )

    def assertRecoveryStateRestoration(
        self,
        error_condition: str,
        expected_state: Dict[str, Any],
        actual_state: Dict[str, Any],
    ) -> None:
        """
        Verify that state was properly restored after recovery.

        Args:
            error_condition: Description of the error condition
            expected_state: Expected state after recovery
            actual_state: Actual state after recovery
        """
        for key, expected_value in expected_state.items():
            self.assertIn(
                key,
                actual_state,
                f"State key '{key}' missing after recovery for error '{error_condition}'",
            )
            self.assertEqual(
                expected_value,
                actual_state[key],
                f"State value mismatch for key '{key}' after recovery for error '{error_condition}': "
                f"expected {expected_value}, got {actual_state[key]}",
            )

    def assertRecoveryActionType(
        self,
        error_condition: str,
        expected_action_type: str,
        actual_action: Union[str, Any],
    ) -> None:
        """
        Verify that the recovery action is of the expected type.

        Args:
            error_condition: Description of the error condition
            expected_action_type: Expected type of recovery action (e.g., 'retry', 'fallback', 'reset')
            actual_action: The actual recovery action taken
        """
        if isinstance(actual_action, str):
            action_str = actual_action.lower()
        else:
            action_str = str(actual_action).lower()

        # Check if action contains expected type indicators
        type_indicators = {
            "retry": ["retry", "again", "reload", "restart"],
            "fallback": ["fallback", "alternative", "backup", "secondary"],
            "reset": ["reset", "restart", "clear", "initialize"],
            "skip": ["skip", "ignore", "continue", "proceed"],
            "wait": ["wait", "sleep", "delay", "pause"],
            "backoff": ["backoff", "exponential", "delay"],
            "circuit_breaker": ["circuit", "breaker", "open", "close"],
            "graceful_degradation": ["degrade", "graceful", "reduce"],
        }

        if expected_action_type in type_indicators:
            indicators = type_indicators[expected_action_type]
            found_indicator = any(
                indicator in action_str for indicator in indicators
            )
            self.assertTrue(
                found_indicator,
                f"Recovery action for error '{error_condition}' doesn't match expected type '{expected_action_type}'. "
                f"Action: {actual_action}",
            )

    def assertRecoveryStrategy(
        self,
        error_condition: str,
        expected_strategy: str,
        actual_trajectory: List[Any],
    ) -> None:
        """
        Verify that the recovery strategy matches the expected approach.

        Args:
            error_condition: Description of the error condition
            expected_strategy: Expected recovery strategy ('immediate', 'delayed', 'progressive')
            actual_trajectory: The actual trajectory from the agent
        """
        # Extract recovery actions
        recovery_actions = []
        for item in actual_trajectory:
            if isinstance(item, dict) and "action" in item:
                action = str(item["action"])
            elif hasattr(item, "action"):
                action = str(item.action)
            elif isinstance(item, str):
                action = item
            else:
                action = str(item)

            if self._is_recovery_action(action):
                recovery_actions.append(action)

        # Analyze recovery strategy
        if expected_strategy == "immediate":
            # Should have recovery actions immediately after error
            self.assertGreater(
                len(recovery_actions),
                0,
                f"No immediate recovery actions found for error '{error_condition}'",
            )
        elif expected_strategy == "delayed":
            # Should have delays before recovery
            delay_indicators = ["wait", "sleep", "delay", "pause"]
            has_delays = any(
                any(
                    indicator in action.lower()
                    for indicator in delay_indicators
                )
                for action in recovery_actions
            )
            self.assertTrue(
                has_delays,
                f"No delayed recovery strategy found for error '{error_condition}'",
            )
        elif expected_strategy == "progressive":
            # Should have multiple recovery attempts with increasing complexity
            self.assertGreater(
                len(recovery_actions),
                1,
                f"No progressive recovery strategy found for error '{error_condition}'",
            )

    def assertRecoverySuccess(
        self,
        error_condition: str,
        success_indicator: Union[str, Callable],
        actual_trajectory: List[Any],
    ) -> None:
        """
        Verify that recovery was successful based on a success indicator.

        Args:
            error_condition: Description of the error condition
            success_indicator: String or function that indicates successful recovery
            actual_trajectory: The actual trajectory from the agent
        """
        if isinstance(success_indicator, str):
            # Check if success indicator appears in trajectory
            trajectory_text = str(actual_trajectory)
            self.assertIn(
                success_indicator,
                trajectory_text,
                f"Recovery success indicator '{success_indicator}' not found for error '{error_condition}'",
            )
        elif callable(success_indicator):
            # Use function to determine success
            success = success_indicator(actual_trajectory)
            self.assertTrue(
                success,
                f"Recovery was not successful for error '{error_condition}'",
            )

    def assertRecoveryFailure(
        self,
        error_condition: str,
        failure_indicator: Union[str, Callable],
        actual_trajectory: List[Any],
    ) -> None:
        """
        Verify that recovery failed based on a failure indicator.

        Args:
            error_condition: Description of the error condition
            failure_indicator: String or function that indicates recovery failure
            actual_trajectory: The actual trajectory from the agent
        """
        if isinstance(failure_indicator, str):
            # Check if failure indicator appears in trajectory
            trajectory_text = str(actual_trajectory)
            self.assertIn(
                failure_indicator,
                trajectory_text,
                f"Recovery failure indicator '{failure_indicator}' not found for error '{error_condition}'",
            )
        elif callable(failure_indicator):
            # Use function to determine failure
            failure = failure_indicator(actual_trajectory)
            self.assertTrue(
                failure, f"Recovery did not fail for error '{error_condition}'"
            )

    def _normalize_action_string(self, action_str: str) -> str:
        """
        Normalize action string for comparison.

        Args:
            action_str: Action string to normalize

        Returns:
            Normalized action string
        """
        normalized = action_str.strip()
        normalized = normalized.replace("'", '"')
        return normalized

    def _find_sequence_in_trajectory(
        self, expected_sequence: List[str], actual_actions: List[str]
    ) -> bool:
        """
        Find if a sequence exists in the trajectory.

        Args:
            expected_sequence: Sequence to find
            actual_actions: List of actual actions

        Returns:
            True if sequence is found, False otherwise
        """
        if len(expected_sequence) > len(actual_actions):
            return False

        for i in range(len(actual_actions) - len(expected_sequence) + 1):
            if (
                actual_actions[i : i + len(expected_sequence)]
                == expected_sequence
            ):
                return True

        return False

    def _is_recovery_action(self, action: Union[str, Any]) -> bool:
        """
        Check if an action is a recovery action.

        Args:
            action: Action to check

        Returns:
            True if action is a recovery action, False otherwise
        """
        if isinstance(action, str):
            action_str = action.lower()
        else:
            action_str = str(action).lower()

        recovery_indicators = [
            "retry",
            "again",
            "reload",
            "refresh",
            "restart",
            "fallback",
            "alternative",
            "backup",
            "reset",
            "clear",
            "skip",
            "ignore",
            "continue",
            "wait",
            "sleep",
            "delay",
            "recover",
            "restore",
            "rollback",
            "undo",
            "revert",
        ]

        return any(
            indicator in action_str for indicator in recovery_indicators
        )
