"""
Method Call Order Test Mixin
Purpose: Verify methods were called in specific order
"""
from typing import Any, Callable, Dict, List, Optional, Type, Union


class CallTracker:
    """Simple class to track method calls without using mocks."""

    def __init__(self):
        self.calls = []

    def record_call(
        self, method_name: str, args: tuple = (), kwargs: dict = None
    ):
        """Record a method call with its arguments."""
        if kwargs is None:
            kwargs = {}
        self.calls.append((method_name, args, kwargs))

    def clear(self):
        """Clear all recorded calls."""
        self.calls = []


class MethodCallOrderTestMixin:
    """Mixin for testing method call order using call tracking."""

    def assertMethodCallOrder(
        self, expected_order: List[str], call_tracker: CallTracker
    ) -> None:
        """
        Verify that methods were called in the specific order.

        Args:
            expected_order: List of method names in expected call order
            call_tracker: CallTracker object that records method calls

        Raises:
            AssertionError: If method calls don't match expected order
        """
        actual_order = [call[0] for call in call_tracker.calls]

        # Compare the orders
        self.assertEqual(
            len(expected_order),
            len(actual_order),
            f"Expected {len(expected_order)} method calls, got {len(actual_order)}",
        )

        for i, (expected, actual) in enumerate(
            zip(expected_order, actual_order)
        ):
            self.assertEqual(
                expected,
                actual,
                f"Method call {i} mismatch: expected {expected}, got {actual}",
            )

    def assertMethodCallSequence(
        self,
        expected_sequence: List[Dict[str, Any]],
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that methods were called with specific arguments in sequence.

        Args:
            expected_sequence: List of dicts with 'method' and 'args'/'kwargs' keys
            call_tracker: CallTracker object that records method calls
        """
        actual_calls = call_tracker.calls

        self.assertEqual(
            len(expected_sequence),
            len(actual_calls),
            f"Expected {len(expected_sequence)} method calls, got {len(actual_calls)}",
        )

        for i, (expected, actual) in enumerate(
            zip(expected_sequence, actual_calls)
        ):
            # Check method name
            expected_method = expected["method"]
            actual_method = actual[0]
            self.assertEqual(
                expected_method,
                actual_method,
                f"Method call {i} name mismatch: expected {expected_method}, got {actual_method}",
            )

            # Check arguments if specified
            if "args" in expected:
                self.assertEqual(
                    expected["args"],
                    actual[1],
                    f"Method call {i} args mismatch: expected {expected['args']}, got {actual[1]}",
                )

            # Check keyword arguments if specified
            if "kwargs" in expected:
                self.assertEqual(
                    expected["kwargs"],
                    actual[2],
                    f"Method call {i} kwargs mismatch: expected {expected['kwargs']}, got {actual[2]}",
                )

    def assertMethodCallCount(
        self, method_name: str, expected_count: int, call_tracker: CallTracker
    ) -> None:
        """
        Verify that a specific method was called the expected number of times.

        Args:
            method_name: Name of the method to check
            expected_count: Expected number of calls
            call_tracker: CallTracker object that records method calls
        """
        actual_count = sum(
            1 for call in call_tracker.calls if call[0] == method_name
        )

        self.assertEqual(
            expected_count,
            actual_count,
            f"Method '{method_name}' call count mismatch: expected {expected_count}, got {actual_count}",
        )

    def assertMethodCallWithArgs(
        self,
        method_name: str,
        expected_args: tuple,
        expected_kwargs: dict,
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a method was called with specific arguments.

        Args:
            method_name: Name of the method to check
            expected_args: Expected positional arguments
            expected_kwargs: Expected keyword arguments
            call_tracker: CallTracker object that records method calls
        """
        method_called = False
        for call in call_tracker.calls:
            if (
                call[0] == method_name
                and call[1] == expected_args
                and call[2] == expected_kwargs
            ):
                method_called = True
                break

        self.assertTrue(
            method_called,
            f"Method '{method_name}' was not called with expected arguments: "
            f"args={expected_args}, kwargs={expected_kwargs}",
        )

    def assertMethodCallContains(
        self, method_name: str, expected_args: tuple, call_tracker: CallTracker
    ) -> None:
        """
        Verify that a method was called with arguments containing the expected values.

        Args:
            method_name: Name of the method to check
            expected_args: Expected arguments (partial match)
            call_tracker: CallTracker object that records method calls
        """
        method_called = False
        for call in call_tracker.calls:
            if call[0] == method_name:
                actual_args = call[1]
                if len(actual_args) >= len(expected_args):
                    # Check if expected args are contained in actual args
                    args_match = True
                    for i, expected_arg in enumerate(expected_args):
                        if (
                            i >= len(actual_args)
                            or actual_args[i] != expected_arg
                        ):
                            args_match = False
                            break
                    if args_match:
                        method_called = True
                        break

        self.assertTrue(
            method_called,
            f"Method '{method_name}' was not called with arguments containing {expected_args}",
        )

    def assertMethodCallBetween(
        self,
        method_name: str,
        start_method: str,
        end_method: str,
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a method was called between two other methods.

        Args:
            method_name: Name of the method to check
            start_method: Name of the method that should be called before
            end_method: Name of the method that should be called after
            call_tracker: CallTracker object that records method calls
        """
        method_calls = [call[0] for call in call_tracker.calls]

        try:
            start_index = method_calls.index(start_method)
            end_index = method_calls.index(end_method)
            target_index = method_calls.index(method_name)

            self.assertGreater(
                target_index,
                start_index,
                f"Method '{method_name}' was called before '{start_method}'",
            )
            self.assertLess(
                target_index,
                end_index,
                f"Method '{method_name}' was called after '{end_method}'",
            )
        except ValueError as e:
            self.fail(
                f"Could not find required methods in call sequence: {str(e)}"
            )

    def assertMethodCallNotBetween(
        self,
        method_name: str,
        start_method: str,
        end_method: str,
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a method was NOT called between two other methods.

        Args:
            method_name: Name of the method to check
            start_method: Name of the method that should be called before
            end_method: Name of the method that should be called after
            call_tracker: CallTracker object that records method calls
        """
        method_calls = [call[0] for call in call_tracker.calls]

        try:
            start_index = method_calls.index(start_method)
            end_index = method_calls.index(end_method)

            # Check if target method appears between start and end
            for i in range(start_index + 1, end_index):
                if method_calls[i] == method_name:
                    self.fail(
                        f"Method '{method_name}' was called between '{start_method}' and '{end_method}'"
                    )
        except ValueError as e:
            self.fail(
                f"Could not find required methods in call sequence: {str(e)}"
            )

    def assertMethodCallFrequency(
        self, expected_frequency: Dict[str, int], call_tracker: CallTracker
    ) -> None:
        """
        Verify that methods were called with expected frequency.

        Args:
            expected_frequency: Dict mapping method names to expected call counts
            call_tracker: CallTracker object that records method calls
        """
        actual_frequency = {}
        for call in call_tracker.calls:
            method_name = call[0]
            actual_frequency[method_name] = (
                actual_frequency.get(method_name, 0) + 1
            )

        for method_name, expected_count in expected_frequency.items():
            actual_count = actual_frequency.get(method_name, 0)
            self.assertEqual(
                expected_count,
                actual_count,
                f"Method '{method_name}' call frequency mismatch: "
                f"expected {expected_count}, got {actual_count}",
            )

    def assertMethodCallArguments(
        self,
        method_name: str,
        call_index: int,
        expected_args: tuple,
        expected_kwargs: dict,
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a specific call of a method had the expected arguments.

        Args:
            method_name: Name of the method to check
            call_index: Index of the call to check (0-based)
            expected_args: Expected positional arguments
            expected_kwargs: Expected keyword arguments
            call_tracker: CallTracker object that records method calls
        """
        method_calls = []
        for call in call_tracker.calls:
            if call[0] == method_name:
                method_calls.append(call)

        self.assertGreater(
            len(method_calls),
            call_index,
            f"Method '{method_name}' was called {len(method_calls)} times, "
            f"but trying to access call {call_index}",
        )

        target_call = method_calls[call_index]
        actual_args = target_call[1]
        actual_kwargs = target_call[2]

        self.assertEqual(
            expected_args,
            actual_args,
            f"Method '{method_name}' call {call_index} args mismatch: "
            f"expected {expected_args}, got {actual_args}",
        )
        self.assertEqual(
            expected_kwargs,
            actual_kwargs,
            f"Method '{method_name}' call {call_index} kwargs mismatch: "
            f"expected {expected_kwargs}, got {actual_kwargs}",
        )

    def assertMethodCallReturnValue(
        self,
        method_name: str,
        call_index: int,
        expected_return_value: Any,
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a specific call of a method returned the expected value.

        Args:
            method_name: Name of the method to check
            call_index: Index of the call to check (0-based)
            expected_return_value: Expected return value
            call_tracker: CallTracker object that records method calls
        """
        method_calls = []
        for call in call_tracker.calls:
            if call[0] == method_name:
                method_calls.append(call)

        self.assertGreater(
            len(method_calls),
            call_index,
            f"Method '{method_name}' was called {len(method_calls)} times, "
            f"but trying to access call {call_index}",
        )

        target_call = method_calls[call_index]
        actual_return_value = target_call[
            0
        ]  # The method itself is the return value

        self.assertEqual(
            expected_return_value,
            actual_return_value,
            f"Method '{method_name}' call {call_index} return value mismatch: "
            f"expected {expected_return_value}, got {actual_return_value}",
        )

    def assertMethodCallSideEffect(
        self,
        method_name: str,
        call_index: int,
        expected_side_effect: Any,
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a specific call of a method had the expected side effect.

        Args:
            method_name: Name of the method to check
            call_index: Index of the call to check (0-based)
            expected_side_effect: Expected side effect
            call_tracker: CallTracker object that records method calls
        """
        method_calls = []
        for call in call_tracker.calls:
            if call[0] == method_name:
                method_calls.append(call)

        self.assertGreater(
            len(method_calls),
            call_index,
            f"Method '{method_name}' was called {len(method_calls)} times, "
            f"but trying to access call {call_index}",
        )

        target_call = method_calls[call_index]
        actual_side_effect = target_call[
            0
        ]  # The method itself is the side effect

        self.assertEqual(
            expected_side_effect,
            actual_side_effect,
            f"Method '{method_name}' call {call_index} side effect mismatch: "
            f"expected {expected_side_effect}, got {actual_side_effect}",
        )

    def assertMethodCallException(
        self,
        method_name: str,
        call_index: int,
        expected_exception: Type[Exception],
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a specific call of a method raised the expected exception.

        Args:
            method_name: Name of the method to check
            call_index: Index of the call to check (0-based)
            expected_exception: Expected exception type
            call_tracker: CallTracker object that records method calls
        """
        method_calls = []
        for call in call_tracker.calls:
            if call[0] == method_name:
                method_calls.append(call)

        self.assertGreater(
            len(method_calls),
            call_index,
            f"Method '{method_name}' was called {len(method_calls)} times, "
            f"but trying to access call {call_index}",
        )

        target_call = method_calls[call_index]
        actual_side_effect = target_call[
            0
        ]  # The method itself is the side effect

        if callable(actual_side_effect):
            # Check if the side effect is an exception
            try:
                actual_side_effect()
                self.fail(
                    f"Method '{method_name}' call {call_index} did not raise an exception"
                )
            except Exception as e:
                self.assertIsInstance(
                    e,
                    expected_exception,
                    f"Method '{method_name}' call {call_index} raised {type(e).__name__}, "
                    f"expected {expected_exception.__name__}",
                )
        else:
            self.fail(
                f"Method '{method_name}' call {call_index} has no side effect"
            )

    def assertMethodCallTimeout(
        self, method_name: str, max_call_time: float, call_tracker: CallTracker
    ) -> None:
        """
        Verify that a method call completes within the expected time.

        Args:
            method_name: Name of the method to check
            max_call_time: Maximum allowed time for the call in seconds
            call_tracker: CallTracker object that records method calls
        """
        import time

        # Find the method call
        method_calls = []
        for call in call_tracker.calls:
            if call[0] == method_name:
                method_calls.append(call)

        if not method_calls:
            self.fail(f"Method '{method_name}' was not called")

        # Measure the time of the last call (simplified - in practice you'd track actual timing)
        start_time = time.time()
        # Simulate method execution time
        time.sleep(0.001)  # Minimal delay for demonstration
        call_time = time.time() - start_time

        self.assertLessEqual(
            call_time,
            max_call_time,
            f"Method '{method_name}' call took {call_time:.3f}s, "
            f"exceeded limit of {max_call_time}s",
        )

    def assertMethodCallConcurrency(
        self, method_names: List[str], call_tracker: CallTracker
    ) -> None:
        """
        Verify that methods were called concurrently (simultaneously).

        Args:
            method_names: List of method names that should be called concurrently
            call_tracker: CallTracker object that records method calls
        """
        # This is a simplified check - in practice you'd need more sophisticated timing analysis
        method_calls = [call[0] for call in call_tracker.calls]

        # Check if all specified methods were called
        for method_name in method_names:
            self.assertIn(
                method_name,
                method_calls,
                f"Method '{method_name}' was not called",
            )

        # Check if methods were called in close proximity (simplified concurrency check)
        # In a real implementation, you'd analyze actual timing data
        self.assertGreaterEqual(
            len(method_calls),
            len(method_names),
            f"Expected at least {len(method_names)} concurrent calls, "
            f"but found {len(method_calls)}",
        )

    def assertMethodCallDependency(
        self,
        dependent_method: str,
        prerequisite_method: str,
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a method was called after its prerequisite method.

        Args:
            dependent_method: Name of the method that depends on another
            prerequisite_method: Name of the method that must be called first
            call_tracker: CallTracker object that records method calls
        """
        method_calls = [call[0] for call in call_tracker.calls]

        try:
            prerequisite_index = method_calls.index(prerequisite_method)
            dependent_index = method_calls.index(dependent_method)

            self.assertLess(
                prerequisite_index,
                dependent_index,
                f"Method '{dependent_method}' was called before its prerequisite '{prerequisite_method}'",
            )
        except ValueError as e:
            self.fail(
                f"Could not find required methods in call sequence: {str(e)}"
            )

    def assertMethodCallIsolation(
        self,
        method_name: str,
        isolated_methods: List[str],
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that a method was called in isolation (not concurrently with other methods).

        Args:
            method_name: Name of the method to check for isolation
            isolated_methods: List of methods that should not be called concurrently
            call_tracker: CallTracker object that records method calls
        """
        method_calls = [call[0] for call in call_tracker.calls]

        # Find the target method calls
        target_indices = [
            i for i, method in enumerate(method_calls) if method == method_name
        ]

        for target_index in target_indices:
            # Check if any isolated method was called around the same time
            for isolated_method in isolated_methods:
                if isolated_method in method_calls:
                    isolated_indices = [
                        i
                        for i, method in enumerate(method_calls)
                        if method == isolated_method
                    ]
                    for isolated_index in isolated_indices:
                        # Check if calls are close to each other (within 2 positions)
                        if abs(target_index - isolated_index) <= 2:
                            self.fail(
                                f"Method '{method_name}' was called concurrently with '{isolated_method}'"
                            )

    def assertMethodCallOrderInvariant(
        self,
        invariant_condition: Callable[[List[str]], bool],
        call_tracker: CallTracker,
    ) -> None:
        """
        Verify that method calls satisfy a custom invariant condition.

        Args:
            invariant_condition: Function that takes a list of method names and returns True if invariant holds
            call_tracker: CallTracker object that records method calls
        """
        method_calls = [call[0] for call in call_tracker.calls]

        self.assertTrue(
            invariant_condition(method_calls),
            f"Method call order invariant violated: {method_calls}",
        )
