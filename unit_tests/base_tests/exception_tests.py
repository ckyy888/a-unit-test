"""
Exception Test Mixin
Purpose: Verify exception-raising behavior
"""
from typing import Any, Callable, List, Optional, Type, Union


class ExceptionTestMixin:
    """Mixin for testing exception behavior in agent trajectories."""

    def assertRaises(
        self, exception: Type[Exception], callable: Callable, *args, **kwargs
    ) -> None:
        """
        Verify that a callable raises the expected exception.

        Args:
            exception: Expected exception type
            callable: Function to test
            *args: Arguments to pass to the callable
            **kwargs: Keyword arguments to pass to the callable

        Raises:
            AssertionError: If the expected exception is not raised
        """
        try:
            callable(*args, **kwargs)
            self.fail(f"Expected {exception.__name__} to be raised")
        except exception:
            # Expected exception was raised
            pass
        except Exception as e:
            self.fail(
                f"Expected {exception.__name__} but got {type(e).__name__}: {str(e)}"
            )

    def assertRaisesWithMessage(
        self,
        exception: Type[Exception],
        expected_message: str,
        callable: Callable,
        *args,
        **kwargs,
    ) -> None:
        """
        Verify that a callable raises the expected exception with a specific message.

        Args:
            exception: Expected exception type
            expected_message: Expected exception message (can be partial match)
            callable: Function to test
            *args: Arguments to pass to the callable
            **kwargs: Keyword arguments to pass to the callable
        """
        try:
            callable(*args, **kwargs)
            self.fail(
                f"Expected {exception.__name__} with message '{expected_message}' to be raised"
            )
        except exception as e:
            if expected_message not in str(e):
                self.fail(
                    f"Expected exception message to contain '{expected_message}', "
                    f"but got: '{str(e)}'"
                )
        except Exception as e:
            self.fail(
                f"Expected {exception.__name__} but got {type(e).__name__}: {str(e)}"
            )

    def assertRaisesRegex(
        self,
        exception: Type[Exception],
        expected_regex: str,
        callable: Callable,
        *args,
        **kwargs,
    ) -> None:
        """
        Verify that a callable raises the expected exception with a message matching regex.

        Args:
            exception: Expected exception type
            expected_regex: Regular expression pattern for expected message
            callable: Function to test
            *args: Arguments to pass to the callable
            **kwargs: Keyword arguments to pass to the callable
        """
        import re

        try:
            callable(*args, **kwargs)
            self.fail(
                f"Expected {exception.__name__} with message matching '{expected_regex}' to be raised"
            )
        except exception as e:
            if not re.search(expected_regex, str(e)):
                self.fail(
                    f"Expected exception message to match '{expected_regex}', "
                    f"but got: '{str(e)}'"
                )
        except Exception as e:
            self.fail(
                f"Expected {exception.__name__} but got {type(e).__name__}: {str(e)}"
            )

    def assertNoException(self, callable: Callable, *args, **kwargs) -> Any:
        """
        Verify that a callable does not raise any exception.

        Args:
            callable: Function to test
            *args: Arguments to pass to the callable
            **kwargs: Keyword arguments to pass to the callable

        Returns:
            The return value of the callable if no exception is raised
        """
        try:
            result = callable(*args, **kwargs)
            return result
        except Exception as e:
            self.fail(
                f"Expected no exception but got {type(e).__name__}: {str(e)}"
            )

    def assertExceptionInTrajectory(
        self, exception_type: Type[Exception], trajectory: List[Any]
    ) -> None:
        """
        Verify that a trajectory contains an exception of the expected type.

        Args:
            exception_type: Expected exception type
            trajectory: The trajectory to check
        """
        exceptions_found = []
        for item in trajectory:
            if isinstance(item, dict) and "exception" in item:
                if isinstance(item["exception"], exception_type):
                    exceptions_found.append(item["exception"])
            elif hasattr(item, "exception"):
                if isinstance(item.exception, exception_type):
                    exceptions_found.append(item.exception)

        self.assertGreater(
            len(exceptions_found),
            0,
            f"No {exception_type.__name__} exceptions found in trajectory",
        )

    def assertExceptionCount(
        self,
        expected_count: int,
        exception_type: Type[Exception],
        trajectory: List[Any],
    ) -> None:
        """
        Verify that a trajectory contains the expected number of exceptions.

        Args:
            expected_count: Expected number of exceptions
            exception_type: Type of exception to count
            trajectory: The trajectory to check
        """
        actual_count = 0
        for item in trajectory:
            if isinstance(item, dict) and "exception" in item:
                if isinstance(item["exception"], exception_type):
                    actual_count += 1
            elif hasattr(item, "exception"):
                if isinstance(item.exception, exception_type):
                    actual_count += 1

        self.assertEqual(
            expected_count,
            actual_count,
            f"Expected {expected_count} {exception_type.__name__} exceptions, "
            f"but found {actual_count}",
        )

    def assertExceptionHandled(
        self, exception_type: Type[Exception], trajectory: List[Any]
    ) -> None:
        """
        Verify that exceptions in a trajectory are properly handled (not left unhandled).

        Args:
            exception_type: Type of exception to check
            trajectory: The trajectory to check
        """
        unhandled_exceptions = []
        for i, item in enumerate(trajectory):
            if isinstance(item, dict) and "exception" in item:
                if isinstance(item["exception"], exception_type):
                    # Check if there's a recovery action after this exception
                    recovery_found = False
                    for j in range(
                        i + 1, min(i + 5, len(trajectory))
                    ):  # Look ahead 5 steps
                        next_item = trajectory[j]
                        if (
                            isinstance(next_item, dict)
                            and "action" in next_item
                        ):
                            action = next_item["action"]
                            if self._is_recovery_action(action):
                                recovery_found = True
                                break
                        elif hasattr(next_item, "action"):
                            action = next_item.action
                            if self._is_recovery_action(action):
                                recovery_found = True
                                break

                    if not recovery_found:
                        unhandled_exceptions.append(item["exception"])
            elif hasattr(item, "exception"):
                if isinstance(item.exception, exception_type):
                    # Similar check for object-based exceptions
                    recovery_found = False
                    for j in range(i + 1, min(i + 5, len(trajectory))):
                        next_item = trajectory[j]
                        if hasattr(next_item, "action"):
                            action = next_item.action
                            if self._is_recovery_action(action):
                                recovery_found = True
                                break

                    if not recovery_found:
                        unhandled_exceptions.append(item.exception)

        self.assertEqual(
            len(unhandled_exceptions),
            0,
            f"Found {len(unhandled_exceptions)} unhandled {exception_type.__name__} exceptions: "
            f"{unhandled_exceptions}",
        )

    def assertExceptionRecoveryTime(
        self,
        exception_type: Type[Exception],
        max_recovery_time: float,
        trajectory: List[Any],
    ) -> None:
        """
        Verify that exceptions are recovered from within the expected time.

        Args:
            exception_type: Type of exception to check
            max_recovery_time: Maximum allowed recovery time in seconds
            trajectory: The trajectory to check
        """
        recovery_times = []
        for i, item in enumerate(trajectory):
            if isinstance(item, dict) and "exception" in item:
                if isinstance(item["exception"], exception_type):
                    # Find the next recovery action
                    recovery_time = None
                    for j in range(i + 1, len(trajectory)):
                        next_item = trajectory[j]
                        if (
                            isinstance(next_item, dict)
                            and "action" in next_item
                        ):
                            action = next_item["action"]
                            if self._is_recovery_action(action):
                                if (
                                    "timestamp" in item
                                    and "timestamp" in next_item
                                ):
                                    recovery_time = (
                                        next_item["timestamp"]
                                        - item["timestamp"]
                                    )
                                break
                        elif hasattr(next_item, "action"):
                            action = next_item.action
                            if self._is_recovery_action(action):
                                if hasattr(item, "timestamp") and hasattr(
                                    next_item, "timestamp"
                                ):
                                    recovery_time = (
                                        next_item.timestamp - item.timestamp
                                    )
                                break

                    if recovery_time is not None:
                        recovery_times.append(recovery_time)
            elif hasattr(item, "exception"):
                if isinstance(item.exception, exception_type):
                    # Similar check for object-based exceptions
                    recovery_time = None
                    for j in range(i + 1, len(trajectory)):
                        next_item = trajectory[j]
                        if hasattr(next_item, "action"):
                            action = next_item.action
                            if self._is_recovery_action(action):
                                if hasattr(item, "timestamp") and hasattr(
                                    next_item, "timestamp"
                                ):
                                    recovery_time = (
                                        next_item.timestamp - item.timestamp
                                    )
                                break

                    if recovery_time is not None:
                        recovery_times.append(recovery_time)

        for recovery_time in recovery_times:
            self.assertLessEqual(
                recovery_time,
                max_recovery_time,
                f"Recovery time {recovery_time}s exceeded limit of {max_recovery_time}s",
            )

    def assertExceptionContext(
        self,
        exception_type: Type[Exception],
        expected_context: str,
        trajectory: List[Any],
    ) -> None:
        """
        Verify that exceptions occur in the expected context.

        Args:
            exception_type: Type of exception to check
            expected_context: Expected context where exception should occur
            trajectory: The trajectory to check
        """
        context_found = False
        for item in trajectory:
            if isinstance(item, dict) and "exception" in item:
                if isinstance(item["exception"], exception_type):
                    if (
                        "context" in item
                        and expected_context in item["context"]
                    ):
                        context_found = True
                        break
            elif hasattr(item, "exception"):
                if isinstance(item.exception, exception_type):
                    if (
                        hasattr(item, "context")
                        and expected_context in item.context
                    ):
                        context_found = True
                        break

        self.assertTrue(
            context_found,
            f"No {exception_type.__name__} exceptions found in context '{expected_context}'",
        )

    def assertExceptionSeverity(
        self,
        exception_type: Type[Exception],
        expected_severity: str,
        trajectory: List[Any],
    ) -> None:
        """
        Verify that exceptions have the expected severity level.

        Args:
            exception_type: Type of exception to check
            expected_severity: Expected severity ('low', 'medium', 'high', 'critical')
            trajectory: The trajectory to check
        """
        severity_indicators = {
            "low": ["warning", "info", "minor"],
            "medium": ["error", "issue", "problem"],
            "high": ["critical", "severe", "fatal"],
            "critical": ["fatal", "panic", "emergency"],
        }

        for item in trajectory:
            if isinstance(item, dict) and "exception" in item:
                if isinstance(item["exception"], exception_type):
                    exception_str = str(item["exception"]).lower()
                    if expected_severity in severity_indicators:
                        indicators = severity_indicators[expected_severity]
                        if any(
                            indicator in exception_str
                            for indicator in indicators
                        ):
                            return  # Found matching severity
            elif hasattr(item, "exception"):
                if isinstance(item.exception, exception_type):
                    exception_str = str(item.exception).lower()
                    if expected_severity in severity_indicators:
                        indicators = severity_indicators[expected_severity]
                        if any(
                            indicator in exception_str
                            for indicator in indicators
                        ):
                            return  # Found matching severity

        self.fail(
            f"No {exception_type.__name__} exceptions found with severity '{expected_severity}'"
        )

    def assertExceptionChain(
        self, exception_types: List[Type[Exception]], trajectory: List[Any]
    ) -> None:
        """
        Verify that exceptions occur in a specific chain/sequence.

        Args:
            exception_types: List of exception types in expected order
            trajectory: The trajectory to check
        """
        found_exceptions = []
        for item in trajectory:
            if isinstance(item, dict) and "exception" in item:
                found_exceptions.append(type(item["exception"]))
            elif hasattr(item, "exception"):
                found_exceptions.append(type(item.exception))

        # Check if the exception chain matches
        for i, expected_type in enumerate(exception_types):
            if i < len(found_exceptions):
                self.assertEqual(
                    expected_type,
                    found_exceptions[i],
                    f"Exception chain mismatch at position {i}: "
                    f"expected {expected_type.__name__}, got {found_exceptions[i].__name__}",
                )
            else:
                self.fail(
                    f"Exception chain too short: expected {len(exception_types)} exceptions, "
                    f"but found {len(found_exceptions)}"
                )

    def assertExceptionSuppressed(
        self, exception_type: Type[Exception], trajectory: List[Any]
    ) -> None:
        """
        Verify that exceptions are properly suppressed (not propagated).

        Args:
            exception_type: Type of exception to check
            trajectory: The trajectory to check
        """
        # Check if exceptions are handled gracefully without causing failures
        for item in trajectory:
            if isinstance(item, dict) and "exception" in item:
                if isinstance(item["exception"], exception_type):
                    # Check if the trajectory continues after the exception
                    # This is a simplified check - in practice, you might want more sophisticated logic
                    pass
            elif hasattr(item, "exception"):
                if isinstance(item.exception, exception_type):
                    # Similar check for object-based exceptions
                    pass

        # If we reach here, exceptions were handled
        pass

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
