"""
Base Test Class
Purpose: Provide high-level assertions and utilities for testing WebArena trajectories
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from unit_tests.action_sequence_tests import ActionSequenceTestMixin
from unit_tests.container_tests import ContainerTestMixin
from unit_tests.exception_tests import ExceptionTestMixin
from unit_tests.method_call_order_tests import MethodCallOrderTestMixin
from unit_tests.recovery_action_tests import RecoveryActionTestMixin
from unit_tests.state_equality_tests import StateEqualityTestMixin
from unit_tests.trajectory_convergence_tests import (
    TrajectoryConvergenceTestMixin,
)


class WebArenaBaseTest(
    unittest.TestCase,
    ActionSequenceTestMixin,
    StateEqualityTestMixin,
    ContainerTestMixin,
    ExceptionTestMixin,
    TrajectoryConvergenceTestMixin,
    RecoveryActionTestMixin,
    MethodCallOrderTestMixin,
):
    """Base test class for WebArena trajectory testing."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.results_dir = Path("shopping_results")
        self.load_test_data()

    def load_test_data(self, results_dir: Optional[Path] = None):
        """
        Load test data from results directory.

        Args:
            results_dir: Optional custom results directory path
        """
        if results_dir:
            self.results_dir = results_dir

        # Load trajectory
        trajectory_file = self.results_dir / "trajectory.json"
        if not trajectory_file.exists():
            raise FileNotFoundError(
                f"Trajectory file not found: {trajectory_file}"
            )

        with open(trajectory_file, "r") as f:
            self.trajectory = json.load(f)

        # Load observations (optional)
        observations_file = self.results_dir / "observations.json"
        if observations_file.exists():
            with open(observations_file, "r") as f:
                self.observations = json.load(f)
        else:
            self.observations = None

        # Load score (optional)
        score_file = self.results_dir / "score.json"
        if score_file.exists():
            with open(score_file, "r") as f:
                self.score = json.load(f)
        else:
            self.score = None

    # High-level assertions for common test cases
    def assertTrajectoryComplete(self, min_steps: int = 6):
        """Assert that trajectory is complete with minimum required steps."""
        self.assertGreaterEqual(
            len(self.trajectory),
            min_steps,
            f"Trajectory should have at least {min_steps} steps",
        )

        # Verify alternating observations and actions
        for i, item in enumerate(self.trajectory):
            if i % 2 == 0:
                self.assertIn(
                    "observation", item, f"Expected observation at index {i}"
                )
                self.assertIn(
                    "text",
                    item["observation"],
                    f"Observation at index {i} should have text",
                )
            else:
                self.assertIn(
                    "action_type", item, f"Expected action at index {i}"
                )

    def assertActionSequenceValid(self, expected_actions: List[int]):
        """Assert that trajectory contains expected action sequence."""
        actions = [
            item
            for item in self.trajectory
            if isinstance(item, dict) and "action_type" in item
        ]
        action_types = [action["action_type"] for action in actions]

        # Verify all expected actions are present in order
        last_idx = -1
        for expected_type in expected_actions:
            try:
                idx = action_types.index(expected_type, last_idx + 1)
                last_idx = idx
            except ValueError:
                self.fail(
                    f"Missing action type {expected_type} after position {last_idx}"
                )

    def assertObservationQuality(self, min_length: int = 100):
        """Assert that observations meet quality standards."""
        for i, item in enumerate(self.trajectory):
            if isinstance(item, dict) and "observation" in item:
                obs_text = item["observation"]["text"]

                # Check observation length
                self.assertGreaterEqual(
                    len(obs_text),
                    min_length,
                    f"Observation {i} should have at least {min_length} characters",
                )

                # Check accessibility tree structure
                self.assertIn(
                    "[", obs_text, f"Observation {i} should have element IDs"
                )
                self.assertIn(
                    "]", obs_text, f"Observation {i} should have element IDs"
                )

    def assertFinalAnswer(self, expected_content: str):
        """Assert that final answer contains expected content."""
        stop_actions = [
            item
            for item in self.trajectory
            if isinstance(item, dict) and item.get("action_type") == 17
        ]

        self.assertEqual(
            1, len(stop_actions), "Should have exactly one stop action"
        )
        stop_action = stop_actions[0]

        self.assertIn(
            "answer", stop_action, "Stop action should have an answer field"
        )
        answer = stop_action["answer"]
        self.assertIn(
            expected_content.lower(),
            answer.lower(),
            f"Answer should contain '{expected_content}'",
        )

    def assertNoErrors(self):
        """Assert that trajectory contains no errors."""
        error_items = []
        for item in self.trajectory:
            if isinstance(item, dict):
                if (
                    "error" in str(item).lower()
                    or (item.get("info", {}).get("fail_error", "") != "")
                    or (item.get("observation", {}).get("error", "") != "")
                ):
                    error_items.append(item)

        self.assertEqual(
            0,
            len(error_items),
            f"Found {len(error_items)} error items in trajectory: {error_items}",
        )

    def assertActionParameters(
        self, action_type: int, required_params: List[str]
    ):
        """Assert that actions of given type have required parameters."""
        actions = [
            item
            for item in self.trajectory
            if isinstance(item, dict)
            and item.get("action_type") == action_type
        ]

        for action in actions:
            for param in required_params:
                self.assertIn(
                    param,
                    action,
                    f"Action type {action_type} missing required parameter '{param}'",
                )

    def assertStateTransitions(
        self, initial_state_pattern: str, final_state_pattern: str
    ):
        """Assert that state transitions from initial to final pattern."""
        pages = []
        for item in self.trajectory:
            if (
                isinstance(item, dict)
                and "observation" in item
                and "info" in item
            ):
                page_info = item["info"].get("page", "unknown")
                pages.append(str(page_info))

        self.assertGreaterEqual(
            len(pages), 2, "Should have at least initial and final states"
        )
        self.assertIn(
            initial_state_pattern,
            str(pages[0]).lower(),
            f"Initial state should contain '{initial_state_pattern}'",
        )
        self.assertIn(
            final_state_pattern,
            str(pages[-1]).lower(),
            f"Final state should contain '{final_state_pattern}'",
        )

    def assertScore(self, expected_score: float, tolerance: float = 0.01):
        """Assert that final score matches expected value within tolerance."""
        if not self.score:
            self.fail("No score file found")

        actual_score = self.score.get("score")
        self.assertIsInstance(
            actual_score, (int, float), "Score should be a number"
        )
        self.assertAlmostEqual(
            expected_score,
            actual_score,
            delta=tolerance,
            msg=f"Score {actual_score} not within {tolerance} of expected {expected_score}",
        )
