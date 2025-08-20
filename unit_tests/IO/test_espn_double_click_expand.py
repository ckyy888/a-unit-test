"""
ESPN Double Click Expand Test Framework
Purpose: Test agent's ability to perform double click to expand team stats row
Tests: DOUBLE CLICK (LEFT) action only
"""
import asyncio
import base64
import json
import logging

# Browser integration imports
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp

sys.path.append("/home/ubuntu/webarena")
from browser_env.actions import ActionTypes, create_double_click_action

# Import base configuration loader
sys.path.append("/home/ubuntu/webarena/unit_tests/IO")


class ConfigLoader:
    """Basic config loader for this test."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            self._config = json.load(f)

    def get(self, key_path: str, default=None):
        keys = key_path.split(".")
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict:
        return self._config.get(section, {})


class EspnDoubleClickValidator:
    """Validator for ESPN double click operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.double_click_executed = False
        self.click_coordinates = None
        self.click_timing = None
        self.team_stats_expanded = False

        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()

    def log_double_click_coordinates_and_timing(
        self, coordinates: Dict, timing: Dict
    ) -> None:
        """Print double-click coordinates and timing."""
        self.click_coordinates = coordinates
        self.click_timing = timing
        self.double_click_executed = True
        print(
            f"Double-click coordinates and timing: ({coordinates['x']}, {coordinates['y']}) with {timing['interval']}ms interval"
        )
        self.logger.info(
            f"Double-click executed at {coordinates} with timing {timing}"
        )

    def validate_double_click_result(self) -> Dict[str, Any]:
        """Validate the double click action result."""
        validation_result = {
            "double_click_executed": self.double_click_executed,
            "coordinates_logged": self.click_coordinates is not None,
            "timing_logged": self.click_timing is not None,
            "team_stats_expanded": self.team_stats_expanded,
            "double_click_successful": False,
            "errors": [],
            "warnings": [],
        }

        # Check if double click was executed
        if not self.double_click_executed:
            validation_result["errors"].append(
                "Double click action was not executed"
            )

        # Check if coordinates were logged
        if not self.click_coordinates:
            validation_result["errors"].append(
                "Double-click coordinates were not logged"
            )

        # Check if timing was logged
        if not self.click_timing:
            validation_result["warnings"].append(
                "Double-click timing was not logged"
            )

        # Check if team stats expanded
        if not self.team_stats_expanded:
            validation_result["errors"].append(
                "Team stats did not expand below row"
            )

        # Determine if double click was successful
        validation_result["double_click_successful"] = (
            self.double_click_executed and self.team_stats_expanded
        )

        # Overall validation status
        validation_result["overall_valid"] = (
            validation_result["double_click_successful"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result


class EspnDoubleClickTestEnvironment:
    """Test environment for ESPN double click with browser-in-browser integration."""

    def __init__(
        self,
        url: str,
        backend_url: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)

        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get(
            "browser_settings.backend_url", "http://localhost:8000"
        )

        # Initialize validator with config
        self.validator = EspnDoubleClickValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 10
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.espn_elements_config = self.config_loader.get_section(
            "espn_double_click_elements"
        )

        self.test_state = {
            "current_step": 0,
            "double_click_executed": False,
            "team_row_collapsed_before": True,
            "team_row_expanded_after": False,
            "detailed_stats_visible": False,
            "action_history": [],
            "screenshots": [],
            "browser_session_active": False,
        }

        # HTTP client for browser backend communication
        self.http_session = None

        # Configure logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_level = self.config_loader.get("logging.level", "INFO")
        log_format = self.config_loader.get(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(
                getattr(logging, log_level.upper(), logging.INFO)
            )
            formatter = logging.Formatter(log_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    async def get_initial_observation(self) -> Dict:
        """Get initial test setup with collapsed team stats row in standings."""
        try:
            # Initialize browser session
            await self._init_browser_session()

            # Navigate to the URL (should show ESPN standings table)
            await self._navigate_to_url()

            # Take initial screenshot
            screenshot_data = await self._get_screenshot()

            # Detect team row and expanded stats elements
            element_detection = await self._detect_standings_elements()
            standings_state = await self._get_current_standings_state()

            # Store initial state
            self.test_state[
                "team_row_collapsed_before"
            ] = not standings_state.get("expanded_stats_visible", False)

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": "initial_collapsed_standings",
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot_data,
                }
            )

            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "Execute double click on team row to expand detailed statistics",
                "element_detection": element_detection,
                "standings_state": standings_state,
                "initial_screenshot": screenshot_data,
                "double_click_elements": self.espn_elements_config,
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_action": "DOUBLE_CLICK(team_row) once only",
                    "expected_result": "detailed team stats expand below row",
                },
                "history": [
                    "ESPN NBA standings table loaded",
                    "Team standings rows are displayed",
                    "Team row is collapsed (basic stats only)",
                    "Detailed stats are hidden",
                    "Double-click will expand team details",
                ],
                "current_standings": {
                    "team_row_collapsed": self.test_state[
                        "team_row_collapsed_before"
                    ],
                    "expanded_stats_visible": False,
                    "double_click_target_ready": True,
                },
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(f"ESPN double click test initialized: {self.url}")
            return setup_result

        except Exception as e:
            self.logger.error(f"Failed to initialize browser session: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def _init_browser_session(self) -> None:
        """Initialize HTTP session and browser backend connection."""
        if not self.http_session:
            self.http_session = aiohttp.ClientSession()

        try:
            async with self.http_session.get(
                f"{self.backend_url}/health"
            ) as response:
                if response.status == 200:
                    self.test_state["browser_session_active"] = True
                    self.logger.info("Browser backend connection established")
                else:
                    raise Exception(
                        f"Browser backend health check failed: {response.status}"
                    )
        except Exception as e:
            self.logger.error(
                f"Cannot connect to browser backend at {self.backend_url}: {e}"
            )
            raise

    async def _navigate_to_url(self) -> Dict:
        """Navigate browser to the test URL."""
        payload = {"url": self.url}

        try:
            async with self.http_session.post(
                f"{self.backend_url}/goto", json=payload
            ) as response:
                result = await response.json()
                if result.get("success"):
                    self.logger.info(f"Successfully navigated to {self.url}")
                    return result
                else:
                    raise Exception(f"Navigation failed: {result}")
        except Exception as e:
            self.logger.error(f"Failed to navigate to URL: {e}")
            raise

    async def _get_screenshot(self) -> str:
        """Get current screenshot from browser backend."""
        try:
            async with self.http_session.get(
                f"{self.backend_url}/screenshot"
            ) as response:
                if response.status == 200:
                    screenshot_bytes = await response.read()
                    screenshot_b64 = base64.b64encode(screenshot_bytes).decode(
                        "utf-8"
                    )
                    return screenshot_b64
                else:
                    raise Exception(
                        f"Screenshot request failed: {response.status}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to get screenshot: {e}")
            raise

    async def _detect_standings_elements(self) -> Dict:
        """Detect team row and expanded stats elements."""
        detected_elements = {}

        for element_type, config in self.espn_elements_config.items():
            detected = {
                "found": False,
                "selector_used": None,
                "coordinates": None,
                "element_info": None,
            }

            # Try each selector to find the element
            for selector in config.get("selectors", []):
                try:
                    eval_script = f"""
                    () => {{
                        const element = document.querySelector('{selector}');
                        if (element) {{
                            const rect = element.getBoundingClientRect();
                            return {{
                                found: true,
                                selector: '{selector}',
                                coordinates: {{
                                    x: rect.left + rect.width / 2,
                                    y: rect.top + rect.height / 2
                                }},
                                visible: rect.width > 0 && rect.height > 0,
                                text_content: element.textContent || '',
                                expandable: element.hasAttribute('data-expandable') || element.classList.contains('expandable'),
                                tag: element.tagName
                            }};
                        }}
                        return {{ found: false }};
                    }}
                    """

                    payload = {"script": eval_script}
                    async with self.http_session.post(
                        f"{self.backend_url}/evaluate", json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if result.get("found"):
                                detected = {
                                    "found": True,
                                    "selector_used": selector,
                                    "coordinates": result.get("coordinates"),
                                    "element_info": result,
                                }
                                break

                except Exception as e:
                    self.logger.warning(
                        f"Failed to check selector {selector}: {e}"
                    )
                    continue

            detected_elements[element_type] = detected

        return detected_elements

    async def _get_current_standings_state(self) -> Dict:
        """Get current standings table and expansion state."""
        try:
            eval_script = """
            () => {
                const result = {
                    expanded_stats_visible: false,
                    team_rows_count: 0,
                    detailed_stats_count: 0
                };

                // Count team rows
                const teamRows = document.querySelectorAll('.team-row, [data-testid="team-standings-row"], .standings-row, tr[data-team]');
                result.team_rows_count = teamRows.length;

                // Check for expanded stats
                const expandedStats = document.querySelectorAll('.expanded-stats, [data-testid="team-details"], .team-stats-detail');
                result.detailed_stats_count = expandedStats.length;

                for (const statsElement of expandedStats) {
                    const rect = statsElement.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        result.expanded_stats_visible = true;
                        break;
                    }
                }

                return result;
            }
            """

            payload = {"script": eval_script}
            async with self.http_session.post(
                f"{self.backend_url}/evaluate", json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to get standings state: {e}")
            return {}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's double click action."""
        self.test_state["current_step"] += 1
        self.test_state["action_history"].append(action)

        evaluation = {
            "step": self.test_state["current_step"],
            "action": action,
            "valid": False,
            "feedback": "",
            "test_completed": False,
            "validation_result": None,
            "browser_result": None,
            "screenshot_after": None,
        }

        # Check step limit (must be exactly 1 step)
        if self.test_state["current_step"] > self.max_steps:
            evaluation[
                "feedback"
            ] = f"Exceeded maximum steps limit ({self.max_steps}) - single action test"
            return evaluation

        try:
            # Execute the action in browser
            browser_result = await self._execute_action_in_browser(action)
            evaluation["browser_result"] = browser_result

            # Log double-click coordinates and timing
            click_info = self._extract_double_click_info(
                action, browser_result
            )
            if click_info:
                self.validator.log_double_click_coordinates_and_timing(
                    click_info["coordinates"], click_info["timing"]
                )

            # Take screenshot after action
            screenshot = await self._get_screenshot()
            evaluation["screenshot_after"] = screenshot

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": self.test_state["current_step"],
                    "action": action.get("action_type"),
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot,
                }
            )

            # Get updated standings state
            await asyncio.sleep(1)  # Wait for expansion animation
            updated_standings_state = await self._get_current_standings_state()
            self.test_state[
                "team_row_expanded_after"
            ] = updated_standings_state.get("expanded_stats_visible", False)
            self.test_state["detailed_stats_visible"] = (
                updated_standings_state.get("detailed_stats_count", 0) > 0
            )

            # Check if team stats expanded
            if self.test_state["team_row_expanded_after"]:
                self.validator.team_stats_expanded = True
                print(f"Detailed team stats expand below row")

            # Evaluate the action
            action_type = action.get("action_type")

            if action_type == 11:  # DOUBLE_CLICK action
                evaluation["valid"] = True
                evaluation["feedback"] = f"Double-click executed on team row"
                self.test_state["double_click_executed"] = True

                if self.test_state["team_row_expanded_after"]:
                    evaluation[
                        "feedback"
                    ] += " - team stats successfully expanded"
                else:
                    evaluation[
                        "feedback"
                    ] += " - team stats expansion not detected"

                evaluation["test_completed"] = True
                evaluation[
                    "validation_result"
                ] = self.validator.validate_double_click_result()
            else:
                evaluation[
                    "feedback"
                ] = f"Unexpected action type {action_type} - expected DOUBLE_CLICK (11) only"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {str(e)}"
            self.logger.error(f"Failed to execute action: {e}")

        return evaluation

    def _extract_double_click_info(
        self, action: Dict, browser_result: Dict
    ) -> Optional[Dict]:
        """Extract double-click coordinates and timing information."""
        # Get coordinates
        if "coordinates" in action:
            coordinates = action["coordinates"]
        else:
            coords = self.espn_elements_config.get("team_row", {}).get(
                "coordinates", {"x": 0.5, "y": 0.4}
            )
            coordinates = coords

        # Get timing information
        timing = {
            "interval": browser_result.get(
                "double_click_interval", 200
            ),  # ms between clicks
            "total_duration": browser_result.get("total_duration", 250),
        }

        return {"coordinates": coordinates, "timing": timing}

    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's double-click action in the browser backend."""
        action_type = action.get("action_type")

        if action_type == 11:  # DOUBLE_CLICK action
            # Determine double-click coordinates
            if "coordinates" in action:
                x, y = action["coordinates"]["x"], action["coordinates"]["y"]
            else:
                coords = self.espn_elements_config.get("team_row", {}).get(
                    "coordinates", {"x": 0.5, "y": 0.4}
                )
                x, y = coords["x"], coords["y"]

            # Execute double-click via browser backend (two rapid clicks)
            payload = {"x": x, "y": y}

            try:
                # First click
                start_time = datetime.now()
                async with self.http_session.post(
                    f"{self.backend_url}/click", json=payload
                ) as response:
                    await response.json()

                # Small delay between clicks (typical double-click timing)
                await asyncio.sleep(0.2)

                # Second click
                async with self.http_session.post(
                    f"{self.backend_url}/click", json=payload
                ) as response:
                    result = await response.json()

                end_time = datetime.now()
                total_duration = (end_time - start_time).total_seconds() * 1000

                result.update(
                    {
                        "double_click_interval": 200,
                        "total_duration": total_duration,
                        "coordinates": {"x": x, "y": y},
                    }
                )

                self.logger.info(f"Double-click executed at ({x}, {y})")
                return result
            except Exception as e:
                self.logger.error(f"Double-click execution failed: {e}")
                raise
        else:
            raise Exception(
                f"Unsupported action type for this test: {action_type}"
            )

    async def cleanup(self) -> None:
        """Cleanup browser session and HTTP connections."""
        if self.http_session:
            await self.http_session.close()
            self.test_state["browser_session_active"] = False
            self.logger.info("Browser session cleanup completed")

    def get_test_status(self) -> Dict:
        """Get current test status."""
        return {
            "current_step": self.test_state["current_step"],
            "max_steps": self.max_steps,
            "double_click_executed": self.test_state["double_click_executed"],
            "team_row_collapsed_before": self.test_state[
                "team_row_collapsed_before"
            ],
            "team_row_expanded_after": self.test_state[
                "team_row_expanded_after"
            ],
            "detailed_stats_visible": self.test_state[
                "detailed_stats_visible"
            ],
            "test_completed": self.test_state["double_click_executed"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps
            - self.test_state["current_step"],
        }

    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            "test_type": "espn_double_click_expand",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "click_coordinates": self.validator.click_coordinates,
            "click_timing": self.validator.click_timing,
            "validation_summary": self.validator.validate_double_click_result(),
            "element_configuration": self.espn_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


# Test runner function for external agent systems
async def test_espn_double_click_expand(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Test function for ESPN double click expand with real browser integration.
    """
    test_env = EspnDoubleClickTestEnvironment(url, backend_url, config_path)

    try:
        initial_obs = await test_env.get_initial_observation()

        if not initial_obs.get("success"):
            return {
                "success": False,
                "error": "Failed to initialize browser environment",
                "details": initial_obs,
                "timestamp": datetime.now().isoformat(),
            }

        agent_actions = run_agent_func(initial_obs)

        if not isinstance(agent_actions, list):
            await test_env.cleanup()
            return {
                "success": False,
                "error": "Agent must return a list of actions",
                "timestamp": datetime.now().isoformat(),
            }

        # Single action test - should only have one action
        if len(agent_actions) != 1:
            await test_env.cleanup()
            return {
                "success": False,
                "error": f"Single action test requires exactly 1 action, got {len(agent_actions)}",
                "timestamp": datetime.now().isoformat(),
            }

        # Evaluate the single action
        evaluation = await test_env.evaluate_agent_action(agent_actions[0])

        test_status = test_env.get_test_status()
        test_report = test_env.get_test_report()

        await test_env.cleanup()

        return {
            "success": evaluation["test_completed"] and evaluation["valid"],
            "initial_observation": initial_obs,
            "agent_actions": agent_actions,
            "evaluations": [evaluation],
            "test_status": test_status,
            "test_report": test_report,
            "browser_integration": True,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        await test_env.cleanup()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# Synchronous wrapper for backward compatibility
def test_espn_double_click_expand_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(
        test_espn_double_click_expand(
            url, run_agent_func, backend_url, config_path
        )
    )


# Convenience function for quick integration
def create_espn_double_click_expand_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> EspnDoubleClickTestEnvironment:
    """Factory function to create an ESPN double click expand test environment."""
    return EspnDoubleClickTestEnvironment(url, backend_url, config_path)
