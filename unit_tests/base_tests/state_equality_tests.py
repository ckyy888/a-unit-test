"""
State Equality Test Mixin
Purpose: Verify state equality and transitions
"""
import copy
import json
from typing import Any, Callable, Dict, List, Optional, Union


class StateEqualityTestMixin:
    """Mixin for testing state equality and transitions."""

    def assertStateEqual(
        self,
        expected_state: Union[Dict[str, Any], str, int],
        actual_state: Union[Dict[str, Any], str, int],
    ) -> None:
        """
        Verify that two states are equal.

        Args:
            expected_state: Expected state (dictionary, string, or integer)
            actual_state: Actual state (dictionary, string, or integer)

        Raises:
            AssertionError: If states are not equal
        """
        if isinstance(expected_state, (str, int)) and isinstance(
            actual_state, (str, int)
        ):
            self.assertEqual(
                expected_state,
                actual_state,
                f"State value mismatch: expected {expected_state}, got {actual_state}",
            )
        elif isinstance(expected_state, dict) and isinstance(
            actual_state, dict
        ):
            self.assertEqual(
                len(expected_state),
                len(actual_state),
                f"State size mismatch: expected {len(expected_state)} keys, got {len(actual_state)}",
            )

            for key, expected_value in expected_state.items():
                self.assertIn(
                    key,
                    actual_state,
                    f"State key '{key}' missing in actual state",
                )
                self.assertEqual(
                    expected_value,
                    actual_state[key],
                    f"State value mismatch for key '{key}': expected {expected_value}, got {actual_state[key]}",
                )
        else:
            self.fail(
                f"Type mismatch: expected {type(expected_state)}, got {type(actual_state)}"
            )

    def assertStateContains(
        self,
        expected_substate: Union[Dict[str, Any], str],
        actual_state: Union[Dict[str, Any], str],
    ) -> None:
        """
        Verify that actual state contains the expected substate.

        Args:
            expected_substate: Expected substate (dictionary or string)
            actual_state: Actual state (dictionary or string)
        """
        if isinstance(expected_substate, str) and isinstance(
            actual_state, str
        ):
            self.assertIn(
                expected_substate,
                actual_state,
                f"Expected substring '{expected_substate}' not found in '{actual_state}'",
            )
        elif isinstance(expected_substate, dict) and isinstance(
            actual_state, dict
        ):
            for key, expected_value in expected_substate.items():
                self.assertIn(
                    key,
                    actual_state,
                    f"State key '{key}' missing in actual state",
                )
                self.assertEqual(
                    expected_value,
                    actual_state[key],
                    f"State value mismatch for key '{key}': expected {expected_value}, got {actual_state[key]}",
                )
        else:
            self.fail(
                f"Type mismatch: expected {type(expected_substate)}, got {type(actual_state)}"
            )

    def assertStateNotEqual(
        self, state1: Dict[str, Any], state2: Dict[str, Any]
    ) -> None:
        """
        Verify that two states are not equal.

        Args:
            state1: First state dictionary
            state2: Second state dictionary
        """
        if state1 == state2:
            self.fail("States are equal but expected them to be different")

    def assertStateTransition(
        self,
        initial_state: Dict[str, Any],
        action: Union[str, Any],
        expected_final_state: Dict[str, Any],
        actual_final_state: Dict[str, Any],
    ) -> None:
        """
        Verify that a state transition occurred correctly.

        Args:
            initial_state: Initial state before action
            action: Action that was performed
            expected_final_state: Expected state after action
            actual_final_state: Actual state after action
        """
        # Verify the state actually changed
        if initial_state == actual_final_state:
            self.fail(f"State did not change after action: {action}")

        # Verify the final state matches expectations
        self.assertStateEqual(expected_final_state, actual_final_state)

    def assertStateInvariant(
        self,
        invariant_condition: Callable[[Dict[str, Any]], bool],
        state: Dict[str, Any],
    ) -> None:
        """
        Verify that a state satisfies a custom invariant condition.

        Args:
            invariant_condition: Function that takes a state and returns True if invariant holds
            state: State to check
        """
        self.assertTrue(
            invariant_condition(state), f"State invariant violated: {state}"
        )

    def assertStateProperty(
        self, property_name: str, expected_value: Any, state: Dict[str, Any]
    ) -> None:
        """
        Verify that a state has a specific property with expected value.

        Args:
            property_name: Name of the property to check
            expected_value: Expected value of the property
            state: State to check
        """
        self.assertIn(
            property_name, state, f"State property '{property_name}' missing"
        )
        self.assertEqual(
            expected_value,
            state[property_name],
            f"State property '{property_name}' value mismatch: "
            f"expected {expected_value}, got {state[property_name]}",
        )

    def assertStatePropertyType(
        self, property_name: str, expected_type: type, state: Dict[str, Any]
    ) -> None:
        """
        Verify that a state property has the expected type.

        Args:
            property_name: Name of the property to check
            expected_type: Expected type of the property
            state: State to check
        """
        self.assertIn(
            property_name, state, f"State property '{property_name}' missing"
        )
        self.assertIsInstance(
            state[property_name],
            expected_type,
            f"State property '{property_name}' has wrong type: "
            f"expected {expected_type.__name__}, got {type(state[property_name]).__name__}",
        )

    def assertStatePropertyRange(
        self,
        property_name: str,
        min_value: Any,
        max_value: Any,
        state: Dict[str, Any],
    ) -> None:
        """
        Verify that a state property is within the expected range.

        Args:
            property_name: Name of the property to check
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            state: State to check
        """
        self.assertIn(
            property_name, state, f"State property '{property_name}' missing"
        )
        actual_value = state[property_name]
        self.assertGreaterEqual(
            actual_value,
            min_value,
            f"State property '{property_name}' value {actual_value} below minimum {min_value}",
        )
        self.assertLessEqual(
            actual_value,
            max_value,
            f"State property '{property_name}' value {actual_value} above maximum {max_value}",
        )

    def assertStatePropertyPattern(
        self, property_name: str, expected_pattern: str, state: Dict[str, Any]
    ) -> None:
        """
        Verify that a state property matches a specific pattern (regex).

        Args:
            property_name: Name of the property to check
            expected_pattern: Regular expression pattern to match
            state: State to check
        """
        import re

        self.assertIn(
            property_name, state, f"State property '{property_name}' missing"
        )
        actual_value = str(state[property_name])
        self.assertIsNotNone(
            re.search(expected_pattern, actual_value),
            f"State property '{property_name}' value '{actual_value}' "
            f"does not match pattern '{expected_pattern}'",
        )

    def assertStatePropertyExists(
        self, property_name: str, state: Dict[str, Any]
    ) -> None:
        """
        Verify that a state property exists (is not None).

        Args:
            property_name: Name of the property to check
            state: State to check
        """
        self.assertIn(
            property_name, state, f"State property '{property_name}' missing"
        )
        self.assertIsNotNone(
            state[property_name], f"State property '{property_name}' is None"
        )

    def assertStatePropertyAbsent(
        self, property_name: str, state: Dict[str, Any]
    ) -> None:
        """
        Verify that a state property does not exist or is None.

        Args:
            property_name: Name of the property to check
            state: State to check
        """
        if property_name in state:
            self.assertIsNone(
                state[property_name],
                f"State property '{property_name}' exists but should be absent",
            )

    def assertStateDeepEqual(
        self, expected_state: Dict[str, Any], actual_state: Dict[str, Any]
    ) -> None:
        """
        Verify that two states are deeply equal (including nested structures).

        Args:
            expected_state: Expected state dictionary
            actual_state: Actual state dictionary
        """
        # Convert to JSON strings for deep comparison
        expected_json = json.dumps(expected_state, sort_keys=True)
        actual_json = json.dumps(actual_state, sort_keys=True)

        self.assertEqual(
            expected_json,
            actual_json,
            f"States are not deeply equal:\nExpected: {expected_state}\nActual: {actual_state}",
        )

    def assertStateSubset(
        self, expected_subset: Dict[str, Any], actual_state: Dict[str, Any]
    ) -> None:
        """
        Verify that actual state contains all key-value pairs from expected subset.

        Args:
            expected_subset: Expected subset of state
            actual_state: Actual state dictionary
        """
        for key, expected_value in expected_subset.items():
            self.assertIn(
                key, actual_state, f"State key '{key}' missing in actual state"
            )
            self.assertEqual(
                expected_value,
                actual_state[key],
                f"State value mismatch for key '{key}': expected {expected_value}, got {actual_state[key]}",
            )

    def assertStateSuperset(
        self, expected_superset: Dict[str, Any], actual_state: Dict[str, Any]
    ) -> None:
        """
        Verify that expected superset contains all key-value pairs from actual state.

        Args:
            expected_superset: Expected superset of state
            actual_state: Actual state dictionary
        """
        for key, actual_value in actual_state.items():
            self.assertIn(
                key,
                expected_superset,
                f"State key '{key}' missing in expected superset",
            )
            self.assertEqual(
                actual_value,
                expected_superset[key],
                f"State value mismatch for key '{key}': expected {expected_superset[key]}, got {actual_value}",
            )

    def assertStateImmutable(
        self,
        state: Dict[str, Any],
        modification_function: Callable[[Dict[str, Any]], None],
    ) -> None:
        """
        Verify that a state is immutable (cannot be modified).

        Args:
            state: State to check for immutability
            modification_function: Function that attempts to modify the state
        """
        original_state = copy.deepcopy(state)

        try:
            modification_function(state)
            # If we reach here, the state was modified
            self.fail("State was modified but should be immutable")
        except (TypeError, AttributeError):
            # Expected - state is immutable
            pass

        # Verify state is unchanged
        self.assertStateEqual(original_state, state)

    def assertStateSerializable(self, state: Dict[str, Any]) -> None:
        """
        Verify that a state can be serialized and deserialized.

        Args:
            state: State to check for serializability
        """
        try:
            # Test JSON serialization
            json_str = json.dumps(state)
            deserialized_state = json.loads(json_str)
            self.assertStateEqual(state, deserialized_state)
        except (TypeError, ValueError) as e:
            self.fail(f"State is not serializable: {str(e)}")

    def assertStateConsistent(
        self,
        consistency_check: Callable[[Dict[str, Any]], bool],
        state: Dict[str, Any],
    ) -> None:
        """
        Verify that a state is internally consistent.

        Args:
            consistency_check: Function that checks state consistency
            state: State to check
        """
        self.assertTrue(
            consistency_check(state),
            f"State consistency check failed: {state}",
        )

    def assertStateTransitionValid(
        self,
        initial_state: Dict[str, Any],
        action: Union[str, Any],
        final_state: Dict[str, Any],
        transition_validator: Callable[
            [Dict[str, Any], Any, Dict[str, Any]], bool
        ],
    ) -> None:
        """
        Verify that a state transition is valid according to a custom validator.

        Args:
            initial_state: Initial state before action
            action: Action that was performed
            final_state: Final state after action
            transition_validator: Function that validates the transition
        """
        self.assertTrue(
            transition_validator(initial_state, action, final_state),
            f"State transition validation failed: {initial_state} -> {action} -> {final_state}",
        )

    def assertStateHistory(
        self,
        expected_history: List[Dict[str, Any]],
        actual_history: List[Dict[str, Any]],
    ) -> None:
        """
        Verify that a sequence of states matches the expected history.

        Args:
            expected_history: Expected sequence of states
            actual_history: Actual sequence of states
        """
        self.assertEqual(
            len(expected_history),
            len(actual_history),
            f"State history length mismatch: expected {len(expected_history)}, got {len(actual_history)}",
        )

        for i, (expected_state, actual_state) in enumerate(
            zip(expected_history, actual_history)
        ):
            self.assertStateEqual(
                expected_state,
                actual_state,
                f"State history mismatch at position {i}",
            )

    def assertStateConvergence(
        self,
        states: List[Dict[str, Any]],
        convergence_condition: Callable[[Dict[str, Any]], bool],
    ) -> None:
        """
        Verify that a sequence of states converges to a condition.

        Args:
            states: Sequence of states to check
            convergence_condition: Function that defines convergence
        """
        # Check if any state satisfies the convergence condition
        converged = any(convergence_condition(state) for state in states)
        self.assertTrue(
            converged,
            f"No state in sequence satisfies convergence condition: {states}",
        )

    def assertStateStability(
        self, states: List[Dict[str, Any]], stability_threshold: int = 3
    ) -> None:
        """
        Verify that states become stable (stop changing).

        Args:
            states: Sequence of states to check
            stability_threshold: Number of consecutive identical states required for stability
        """
        if len(states) < stability_threshold:
            self.fail(
                f"Not enough states to check stability: {len(states)} < {stability_threshold}"
            )

        # Check if the last few states are identical
        last_states = states[-stability_threshold:]
        for i in range(1, len(last_states)):
            self.assertStateEqual(
                last_states[0],
                last_states[i],
                f"States are not stable: {last_states}",
            )

    def assertStateBounded(
        self,
        states: List[Dict[str, Any]],
        bound_check: Callable[[Dict[str, Any]], bool],
    ) -> None:
        """
        Verify that all states in a sequence satisfy a bound condition.

        Args:
            states: Sequence of states to check
            bound_check: Function that defines the bound condition
        """
        for i, state in enumerate(states):
            self.assertTrue(
                bound_check(state),
                f"State at position {i} violates bound condition: {state}",
            )

    def assertStateMonotonic(
        self,
        states: List[Dict[str, Any]],
        monotonic_check: Callable[[Dict[str, Any], Dict[str, Any]], bool],
    ) -> None:
        """
        Verify that a sequence of states is monotonic (each state relates to the previous one).

        Args:
            states: Sequence of states to check
            monotonic_check: Function that checks monotonicity between two states
        """
        for i in range(1, len(states)):
            self.assertTrue(
                monotonic_check(states[i - 1], states[i]),
                f"States at positions {i-1} and {i} violate monotonicity: {states[i-1]} -> {states[i]}",
            )

    def assertStateReachable(
        self,
        target_state: Dict[str, Any],
        reachable_states: List[Dict[str, Any]],
    ) -> None:
        """
        Verify that a target state is reachable from a set of states.

        Args:
            target_state: Target state to check
            reachable_states: List of states that should be reachable
        """
        # Check if target state is in the reachable set
        for reachable_state in reachable_states:
            if self._states_equal(target_state, reachable_state):
                return  # Found matching state

        self.fail(f"Target state is not reachable: {target_state}")

    def assertStateUnreachable(
        self,
        target_state: Dict[str, Any],
        reachable_states: List[Dict[str, Any]],
    ) -> None:
        """
        Verify that a target state is NOT reachable from a set of states.

        Args:
            target_state: Target state to check
            reachable_states: List of states that should be reachable
        """
        # Check if target state is in the reachable set
        for reachable_state in reachable_states:
            if self._states_equal(target_state, reachable_state):
                self.fail(
                    f"Target state should not be reachable: {target_state}"
                )

    def _states_equal(
        self, state1: Dict[str, Any], state2: Dict[str, Any]
    ) -> bool:
        """
        Check if two states are equal.

        Args:
            state1: First state
            state2: Second state

        Returns:
            True if states are equal, False otherwise
        """
        try:
            return json.dumps(state1, sort_keys=True) == json.dumps(
                state2, sort_keys=True
            )
        except (TypeError, ValueError):
            # Fallback to direct comparison
            return state1 == state2
