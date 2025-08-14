"""
Trajectory Convergence Test Mixin
Purpose: Verify trajectory convergence behavior
"""
from typing import Any, Callable, Dict, List, Optional, Union
import math


class TrajectoryConvergenceTestMixin:
    """Mixin for testing trajectory convergence behavior."""
    
    def assertTrajectoryConverges(self, trajectory: List[Any], 
                                convergence_condition: Callable[[Any], bool], 
                                max_steps: Optional[int] = None) -> None:
        """
        Verify that a trajectory converges to a condition within a maximum number of steps.
        
        Args:
            trajectory: The trajectory to check
            convergence_condition: Function that returns True when convergence is reached
            max_steps: Maximum number of steps allowed for convergence (None for no limit)
        """
        converged = False
        convergence_step = None
        
        for i, step in enumerate(trajectory):
            if convergence_condition(step):
                converged = True
                convergence_step = i
                break
        
        self.assertTrue(converged,
                       f"Trajectory did not converge within {len(trajectory)} steps")
        
        if max_steps is not None:
            self.assertLessEqual(convergence_step, max_steps,
                               f"Trajectory converged at step {convergence_step}, "
                               f"but should converge within {max_steps} steps")
    
    def assertTrajectoryConvergesTo(self, trajectory: List[Any], 
                                  expected_final_state: Any, 
                                  tolerance: float = 1e-6) -> None:
        """
        Verify that a trajectory converges to a specific final state.
        
        Args:
            trajectory: The trajectory to check
            expected_final_state: Expected final state
            tolerance: Tolerance for numerical comparisons
        """
        if not trajectory:
            self.fail("Trajectory is empty")
        
        final_state = trajectory[-1]
        self._assert_states_equal(final_state, expected_final_state, tolerance)
    
    def assertTrajectoryConvergesWithin(self, trajectory: List[Any], 
                                      convergence_condition: Callable[[Any], bool], 
                                      max_steps: int) -> None:
        """
        Verify that a trajectory converges within a specific number of steps.
        
        Args:
            trajectory: The trajectory to check
            convergence_condition: Function that returns True when convergence is reached
            max_steps: Maximum number of steps allowed for convergence
        """
        converged = False
        convergence_step = None
        
        for i, step in enumerate(trajectory):
            if convergence_condition(step):
                converged = True
                convergence_step = i
                break
        
        self.assertTrue(converged,
                       f"Trajectory did not converge within {len(trajectory)} steps")
        self.assertLessEqual(convergence_step, max_steps,
                           f"Trajectory converged at step {convergence_step}, "
                           f"but should converge within {max_steps} steps")
    
    def assertTrajectoryMonotonicConvergence(self, trajectory: List[Any], 
                                           monotonic_condition: Callable[[Any, Any], bool], 
                                           convergence_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory converges monotonically to a condition.
        
        Args:
            trajectory: The trajectory to check
            monotonic_condition: Function that checks if step i+1 is better than step i
            convergence_condition: Function that returns True when convergence is reached
        """
        # Check monotonicity
        for i in range(1, len(trajectory)):
            self.assertTrue(monotonic_condition(trajectory[i-1], trajectory[i]),
                          f"Trajectory is not monotonic at step {i}")
        
        # Check convergence
        converged = any(convergence_condition(step) for step in trajectory)
        self.assertTrue(converged,
                       f"Trajectory does not converge to the specified condition")
    
    def assertTrajectoryBoundedConvergence(self, trajectory: List[Any], 
                                         bound_condition: Callable[[Any], bool], 
                                         convergence_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory converges while staying within bounds.
        
        Args:
            trajectory: The trajectory to check
            bound_condition: Function that returns True if a step is within bounds
            convergence_condition: Function that returns True when convergence is reached
        """
        # Check that all steps are within bounds
        for i, step in enumerate(trajectory):
            self.assertTrue(bound_condition(step),
                          f"Trajectory step {i} violates bounds: {step}")
        
        # Check convergence
        converged = any(convergence_condition(step) for step in trajectory)
        self.assertTrue(converged,
                       f"Trajectory does not converge to the specified condition")
    
    def assertTrajectoryRateConvergence(self, trajectory: List[Any], 
                                      rate_condition: Callable[[Any, Any], bool], 
                                      convergence_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory converges at a specific rate.
        
        Args:
            trajectory: The trajectory to check
            rate_condition: Function that checks if convergence rate is acceptable
            convergence_condition: Function that returns True when convergence is reached
        """
        # Check convergence rate
        for i in range(1, len(trajectory)):
            self.assertTrue(rate_condition(trajectory[i-1], trajectory[i]),
                          f"Trajectory convergence rate violated at step {i}")
        
        # Check convergence
        converged = any(convergence_condition(step) for step in trajectory)
        self.assertTrue(converged,
                       f"Trajectory does not converge to the specified condition")
    
    def assertTrajectoryLinearConvergence(self, trajectory: List[Any], 
                                        tolerance: float = 1e-6) -> None:
        """
        Verify that a trajectory converges linearly.
        
        Args:
            trajectory: The trajectory to check
            tolerance: Tolerance for numerical comparisons
        """
        if len(trajectory) < 3:
            self.fail("Trajectory too short to check linear convergence")
        
        # Check if trajectory can be represented as numerical values
        try:
            values = [float(step) if isinstance(step, (int, float)) else step for step in trajectory]
        except (ValueError, TypeError):
            self.fail("Trajectory steps must be convertible to numerical values for linear convergence check")
        
        # Calculate convergence rate
        for i in range(2, len(values)):
            if abs(values[i-1]) > tolerance:
                rate1 = abs(values[i] - values[i-1]) / abs(values[i-1])
                rate2 = abs(values[i-1] - values[i-2]) / abs(values[i-2])
                
                # Check if rates are approximately equal (linear convergence)
                self.assertAlmostEqual(rate1, rate2, delta=tolerance,
                                     msg=f"Linear convergence rate violated at step {i}")
    
    def assertTrajectoryQuadraticConvergence(self, trajectory: List[Any], 
                                           tolerance: float = 1e-6) -> None:
        """
        Verify that a trajectory converges quadratically.
        
        Args:
            trajectory: The trajectory to check
            tolerance: Tolerance for numerical comparisons
        """
        if len(trajectory) < 3:
            self.fail("Trajectory too short to check quadratic convergence")
        
        # Check if trajectory can be represented as numerical values
        try:
            values = [float(step) if isinstance(step, (int, float)) else step for step in trajectory]
        except (ValueError, TypeError):
            self.fail("Trajectory steps must be convertible to numerical values for quadratic convergence check")
        
        # Calculate convergence rate
        for i in range(2, len(values)):
            if abs(values[i-1]) > tolerance:
                rate1 = abs(values[i] - values[i-1]) / (abs(values[i-1]) ** 2)
                rate2 = abs(values[i-1] - values[i-2]) / (abs(values[i-2]) ** 2)
                
                # Check if rates are approximately equal (quadratic convergence)
                self.assertAlmostEqual(rate1, rate2, delta=tolerance,
                                     msg=f"Quadratic convergence rate violated at step {i}")
    
    def assertTrajectoryExponentialConvergence(self, trajectory: List[Any], 
                                             base: float = 0.5, 
                                             tolerance: float = 1e-6) -> None:
        """
        Verify that a trajectory converges exponentially.
        
        Args:
            trajectory: The trajectory to check
            base: Expected base for exponential convergence
            tolerance: Tolerance for numerical comparisons
        """
        if len(trajectory) < 3:
            self.fail("Trajectory too short to check exponential convergence")
        
        # Check if trajectory can be represented as numerical values
        try:
            values = [float(step) if isinstance(step, (int, float)) else step for step in trajectory]
        except (ValueError, TypeError):
            self.fail("Trajectory steps must be convertible to numerical values for exponential convergence check")
        
        # Calculate convergence rate
        for i in range(1, len(values)):
            if abs(values[i-1]) > tolerance:
                rate = abs(values[i]) / abs(values[i-1])
                self.assertAlmostEqual(rate, base, delta=tolerance,
                                     msg=f"Exponential convergence rate violated at step {i}")
    
    def assertTrajectoryStableConvergence(self, trajectory: List[Any], 
                                        stability_threshold: int = 3, 
                                        tolerance: float = 1e-6) -> None:
        """
        Verify that a trajectory converges to a stable state.
        
        Args:
            trajectory: The trajectory to check
            stability_threshold: Number of consecutive stable steps required
            tolerance: Tolerance for stability comparison
        """
        if len(trajectory) < stability_threshold:
            self.fail(f"Trajectory too short to check stability: {len(trajectory)} < {stability_threshold}")
        
        # Check if the last few steps are stable
        last_steps = trajectory[-stability_threshold:]
        for i in range(1, len(last_steps)):
            self._assert_states_equal(last_steps[i-1], last_steps[i], tolerance,
                                    f"Trajectory is not stable at step {len(trajectory) - stability_threshold + i}")
    
    def assertTrajectoryOscillatoryConvergence(self, trajectory: List[Any], 
                                             oscillation_condition: Callable[[Any, Any], bool], 
                                             convergence_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory converges with oscillations.
        
        Args:
            trajectory: The trajectory to check
            oscillation_condition: Function that checks if two consecutive steps oscillate
            convergence_condition: Function that returns True when convergence is reached
        """
        # Check for oscillations
        oscillations_found = False
        for i in range(1, len(trajectory)):
            if oscillation_condition(trajectory[i-1], trajectory[i]):
                oscillations_found = True
                break
        
        self.assertTrue(oscillations_found,
                       "Trajectory does not exhibit oscillations")
        
        # Check convergence
        converged = any(convergence_condition(step) for step in trajectory)
        self.assertTrue(converged,
                       f"Trajectory does not converge to the specified condition")
    
    def assertTrajectoryCyclicConvergence(self, trajectory: List[Any], 
                                        cycle_length: int, 
                                        convergence_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory converges with cyclic behavior.
        
        Args:
            trajectory: The trajectory to check
            cycle_length: Expected length of the cycle
            convergence_condition: Function that returns True when convergence is reached
        """
        if len(trajectory) < cycle_length * 2:
            self.fail(f"Trajectory too short to check cyclic behavior: {len(trajectory)} < {cycle_length * 2}")
        
        # Check for cyclic behavior
        for i in range(cycle_length, len(trajectory)):
            self.assertEqual(trajectory[i], trajectory[i - cycle_length],
                           f"Trajectory does not exhibit cyclic behavior with length {cycle_length}")
        
        # Check convergence
        converged = any(convergence_condition(step) for step in trajectory)
        self.assertTrue(converged,
                       f"Trajectory does not converge to the specified condition")
    
    def assertTrajectoryChaoticConvergence(self, trajectory: List[Any], 
                                         chaos_condition: Callable[[Any, Any], bool], 
                                         convergence_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory converges with chaotic behavior.
        
        Args:
            trajectory: The trajectory to check
            chaos_condition: Function that checks if two consecutive steps exhibit chaotic behavior
            convergence_condition: Function that returns True when convergence is reached
        """
        # Check for chaotic behavior
        chaotic_steps = 0
        for i in range(1, len(trajectory)):
            if chaos_condition(trajectory[i-1], trajectory[i]):
                chaotic_steps += 1
        
        self.assertGreater(chaotic_steps, 0,
                          "Trajectory does not exhibit chaotic behavior")
        
        # Check convergence
        converged = any(convergence_condition(step) for step in trajectory)
        self.assertTrue(converged,
                       f"Trajectory does not converge to the specified condition")
    
    def assertTrajectoryDivergence(self, trajectory: List[Any], 
                                 divergence_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory diverges according to a condition.
        
        Args:
            trajectory: The trajectory to check
            divergence_condition: Function that returns True when divergence is detected
        """
        diverged = any(divergence_condition(step) for step in trajectory)
        self.assertTrue(diverged,
                       f"Trajectory does not diverge according to the specified condition")
    
    def assertTrajectoryBoundedDivergence(self, trajectory: List[Any], 
                                        divergence_condition: Callable[[Any], bool], 
                                        bound_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory diverges while staying within bounds.
        
        Args:
            trajectory: The trajectory to check
            divergence_condition: Function that returns True when divergence is detected
            bound_condition: Function that returns True if a step is within bounds
        """
        # Check that all steps are within bounds
        for i, step in enumerate(trajectory):
            self.assertTrue(bound_condition(step),
                          f"Trajectory step {i} violates bounds: {step}")
        
        # Check divergence
        diverged = any(divergence_condition(step) for step in trajectory)
        self.assertTrue(diverged,
                       f"Trajectory does not diverge according to the specified condition")
    
    def assertTrajectoryLimitCycle(self, trajectory: List[Any], 
                                 cycle_condition: Callable[[List[Any]], bool]) -> None:
        """
        Verify that a trajectory converges to a limit cycle.
        
        Args:
            trajectory: The trajectory to check
            cycle_condition: Function that checks if a subsequence forms a limit cycle
        """
        # Check for limit cycle behavior
        for i in range(len(trajectory) // 2, len(trajectory)):
            subsequence = trajectory[i:]
            if cycle_condition(subsequence):
                return  # Found limit cycle
        
        self.fail("Trajectory does not converge to a limit cycle")
    
    def assertTrajectoryAttractor(self, trajectory: List[Any], 
                                attractor_condition: Callable[[Any], bool], 
                                basin_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory converges to an attractor.
        
        Args:
            trajectory: The trajectory to check
            attractor_condition: Function that returns True if a step is in the attractor
            basin_condition: Function that returns True if a step is in the basin of attraction
        """
        # Check that trajectory starts in basin of attraction
        self.assertTrue(basin_condition(trajectory[0]),
                       "Trajectory does not start in basin of attraction")
        
        # Check that trajectory converges to attractor
        converged_to_attractor = any(attractor_condition(step) for step in trajectory)
        self.assertTrue(converged_to_attractor,
                       "Trajectory does not converge to attractor")
    
    def assertTrajectoryLyapunovStable(self, trajectory: List[Any], 
                                     lyapunov_condition: Callable[[Any, Any], bool]) -> None:
        """
        Verify that a trajectory is Lyapunov stable.
        
        Args:
            trajectory: The trajectory to check
            lyapunov_condition: Function that checks Lyapunov stability between two steps
        """
        # Check Lyapunov stability
        for i in range(1, len(trajectory)):
            self.assertTrue(lyapunov_condition(trajectory[i-1], trajectory[i]),
                          f"Trajectory is not Lyapunov stable at step {i}")
    
    def assertTrajectoryAsymptoticallyStable(self, trajectory: List[Any], 
                                           asymptotic_condition: Callable[[Any, Any], bool]) -> None:
        """
        Verify that a trajectory is asymptotically stable.
        
        Args:
            trajectory: The trajectory to check
            asymptotic_condition: Function that checks asymptotic stability between two steps
        """
        # Check asymptotic stability
        for i in range(1, len(trajectory)):
            self.assertTrue(asymptotic_condition(trajectory[i-1], trajectory[i]),
                          f"Trajectory is not asymptotically stable at step {i}")
    
    def assertTrajectoryInvariant(self, trajectory: List[Any], 
                                invariant_condition: Callable[[Any], bool]) -> None:
        """
        Verify that a trajectory satisfies an invariant condition.
        
        Args:
            trajectory: The trajectory to check
            invariant_condition: Function that returns True if invariant holds
        """
        # Check invariant at each step
        for i, step in enumerate(trajectory):
            self.assertTrue(invariant_condition(step),
                          f"Trajectory invariant violated at step {i}: {step}")
    
    def assertTrajectoryConvergenceRate(self, trajectory: List[Any], 
                                      expected_rate: float, 
                                      tolerance: float = 1e-6) -> None:
        """
        Verify that a trajectory converges at a specific rate.
        
        Args:
            trajectory: The trajectory to check
            expected_rate: Expected convergence rate
            tolerance: Tolerance for rate comparison
        """
        if len(trajectory) < 2:
            self.fail("Trajectory too short to check convergence rate")
        
        # Calculate actual convergence rate
        try:
            values = [float(step) if isinstance(step, (int, float)) else step for step in trajectory]
        except (ValueError, TypeError):
            self.fail("Trajectory steps must be convertible to numerical values for rate check")
        
        # Calculate rate as ratio of consecutive differences
        rates = []
        for i in range(2, len(values)):
            if abs(values[i-1] - values[i-2]) > tolerance:
                rate = abs(values[i] - values[i-1]) / abs(values[i-1] - values[i-2])
                rates.append(rate)
        
        if rates:
            avg_rate = sum(rates) / len(rates)
            self.assertAlmostEqual(avg_rate, expected_rate, delta=tolerance,
                                 msg=f"Convergence rate mismatch: expected {expected_rate}, got {avg_rate}")
    
    def _assert_states_equal(self, state1: Any, state2: Any, tolerance: float = 1e-6, 
                           message: str = "") -> None:
        """
        Assert that two states are equal within tolerance.
        
        Args:
            state1: First state
            state2: Second state
            tolerance: Tolerance for numerical comparisons
            message: Custom error message
        """
        if isinstance(state1, (int, float)) and isinstance(state2, (int, float)):
            self.assertAlmostEqual(state1, state2, delta=tolerance, msg=message)
        else:
            self.assertEqual(state1, state2, msg=message) 