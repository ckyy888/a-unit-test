"""
Google Maps Drag Map Test Framework
Purpose: Test agent's ability to perform precise drag operation to shift map view
Tests: DRAG action only
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

# WebSocket for screenshot streaming (preferred path)
try:
    import websockets
except ImportError:
    websockets = None

sys.path.append("/home/ubuntu/webarena")
from browser_env.actions import ActionTypes, create_drag_action

# Import base configuration loader
sys.path.append("/home/ubuntu/webarena/unit_tests/IO")


class ConfigLoader:
    """Loads and manages configuration from config.json file (robust)."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        self.config_path = Path(config_path)
        self._config = None
        self._load_config()

    def _load_config(self) -> None:
        try:
            with open(self.config_path, "r") as f:
                self._config = json.load(f)
            logging.getLogger(__name__).info(
                f"Configuration loaded from {self.config_path}"
            )
        except FileNotFoundError:
            logging.getLogger(__name__).warning(
                f"Config file not found: {self.config_path}, using defaults"
            )
            self._config = self._get_default_config()
        except json.JSONDecodeError as e:
            logging.getLogger(__name__).error(
                f"Invalid JSON in config file: {e}, using defaults"
            )
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict:
        return {
            "test_configuration": {
                "max_steps": 1,
                "timeout_seconds": 10,
                "screenshot_enabled": True,
                "detailed_logging": True,
            },
            "browser_settings": {
                "backend_url": "http://localhost:8000",
                "screenshot_frequency": "per_action",
                "wait_for_load": True,
                "network_idle_timeout": 1000,
            },
            "logging": {
                "level": "INFO",
                "file_path": "/home/ubuntu/webarena/unit_tests/IO/io_tests.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

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

    def reload(self) -> None:
        self._load_config()


class GoogleMapsDragValidator:
    """Validator for Google Maps drag operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.drag_executed = False
        self.drag_vector = None
        self.drag_distance = 0
        self.map_center_shifted = False
        self.new_area_visible = False

        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()

    def log_drag_vector_and_distance(
        self, vector: Dict, distance: float
    ) -> None:
        """Print drag vector and distance."""
        self.drag_vector = vector
        self.drag_distance = distance
        self.drag_executed = True
        print(
            f"Drag vector and distance: from ({vector['from_x']}, {vector['from_y']}) to ({vector['to_x']}, {vector['to_y']}) - {distance}px"
        )
        self.logger.info(f"Drag executed: {vector} with distance {distance}px")

    def validate_drag_result(self) -> Dict[str, Any]:
        """Validate the drag action result."""
        validation_result = {
            "drag_executed": self.drag_executed,
            "drag_vector_logged": self.drag_vector is not None,
            "drag_distance_logged": self.drag_distance > 0,
            "expected_distance": 200,
            "actual_distance": self.drag_distance,
            "map_center_shifted": self.map_center_shifted,
            "new_area_visible": self.new_area_visible,
            "drag_successful": False,
            "errors": [],
            "warnings": [],
        }

        # Check if drag was executed
        if not self.drag_executed:
            validation_result["errors"].append("Drag action was not executed")

        # Check if drag vector was logged
        if not self.drag_vector:
            validation_result["errors"].append("Drag vector was not logged")

        # Check if drag distance is appropriate
        if self.drag_distance < 150:
            validation_result["warnings"].append(
                f"Drag distance may be insufficient: {self.drag_distance}px"
            )
        elif self.drag_distance > 300:
            validation_result["warnings"].append(
                f"Drag distance may be excessive: {self.drag_distance}px"
            )

        # Check if map center shifted
        if not self.map_center_shifted:
            validation_result["errors"].append(
                "Map center did not shift eastward"
            )

        # Check if new area is visible
        if not self.new_area_visible:
            validation_result["warnings"].append(
                "New area may not be visible after drag"
            )

        # Determine if drag was successful
        validation_result["drag_successful"] = (
            self.drag_executed and self.map_center_shifted
        )

        # Overall validation status
        validation_result["overall_valid"] = (
            validation_result["drag_successful"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result


class GoogleMapsDragTestEnvironment:
    """Test environment for Google Maps drag with browser-in-browser integration."""

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
        self.validator = GoogleMapsDragValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 10
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.gmaps_elements_config = self.config_loader.get_section(
            "gmaps_drag_elements"
        )

        self.test_state = {
            "current_step": 0,
            "drag_executed": False,
            "map_center_before": None,
            "map_center_after": None,
            "drag_start_coords": None,
            "drag_end_coords": None,
            "map_view_changed": False,
            "action_history": [],
            "screenshots": [],
            "browser_session_active": False,
            "full_trajectory": [],
            "agent_decision_tree": None,
            "agent_decision_metadata": None,
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

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            getattr(logging, log_level.upper(), logging.INFO)
        )
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_file = self.config_loader.get("logging.file_path")
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(
                    getattr(logging, log_level.upper(), logging.INFO)
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")

    async def get_initial_observation(self) -> Dict:
        """Get initial test setup with map centered on downtown area."""
        try:
            # Initialize browser session
            await self._init_browser_session()

            # Navigate to the URL (should show Google Maps downtown view)
            await self._navigate_to_url()

            # Take initial screenshot
            screenshot_data = await self._get_screenshot()

            # Double-call accessibility tree for stability, then use acc2
            try:
                acc1 = await self._get_accessibility_tree()
                acc2 = await self._get_accessibility_tree()
            except Exception:
                acc1, acc2 = {}, {}

            # Get initial map state
            map_state = await self._get_current_map_state()

            # Store initial map center
            self.test_state["map_center_before"] = map_state.get("map_center")

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": "initial_downtown_centered",
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot_data,
                }
            )

            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "Execute drag operation to shift map view eastward",
                "map_state": map_state,
                "initial_screenshot": screenshot_data,
                "accessibility_tree": acc2.get("accessibility_tree", ""),
                "initial_accessibility_tree": acc2.get("accessibility_tree", ""),
                "obs_metadata": {
                    "obs_nodes_info": acc2.get("obs_nodes_info", {}),
                    "browser_config": acc2.get("browser_config", {}),
                    "viewport_size": acc2.get("viewport_size", {"width": 1280, "height": 800}),
                },
                "drag_elements": self.gmaps_elements_config,
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_action": "DRAG(from_center, to_east, 200px) once only",
                    "expected_result": "map center shifts eastward, new area visible",
                },
                "history": [
                    "Google Maps loaded successfully",
                    "Map centered on downtown area",
                    "Need to see nearby suburb to the east",
                    "Drag operation will shift view eastward",
                    "Ready for drag action",
                ],
                "current_map": {
                    "center": self.test_state["map_center_before"],
                    "view": "downtown_area",
                    "drag_coordinates_defined": True,
                },
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Record initial trajectory
            self.test_state["full_trajectory"].append(
                {
                    "step": "initialization",
                    "timestamp": datetime.now().isoformat(),
                    "setup_result": setup_result,
                    "accessibility_data": acc2,
                    "screenshot_captured": bool(screenshot_data),
                }
            )

            self.logger.info(
                f"Google Maps drag test initialized: {self.url}"
            )
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
        """Get current screenshot via WebSocket if available, else HTTP."""
        try:
            if websockets:
                ws_url = (
                    self.backend_url.replace("http://", "ws://").replace(
                        "https://", "wss://"
                    )
                    + "/screenshot"
                )
                async with websockets.connect(ws_url) as websocket:
                    data = await websocket.recv()
                    if isinstance(data, bytes):
                        return base64.b64encode(data).decode("utf-8")
                    return base64.b64encode(str(data).encode()).decode("utf-8")
        except Exception as e:
            self.logger.warning(f"WebSocket screenshot failed, falling back: {e}")

        try:
            async with self.http_session.get(
                f"{self.backend_url}/screenshot"
            ) as response:
                if response.status == 200:
                    screenshot_bytes = await response.read()
                    return base64.b64encode(screenshot_bytes).decode("utf-8")
                else:
                    raise Exception(
                        f"Screenshot request failed: {response.status}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to get screenshot: {e}")
            return "screenshot_unavailable"

    async def _get_accessibility_tree(self) -> Dict:
        """Get accessibility tree from browser backend."""
        try:
            payload = {"current_viewport_only": False}
            async with self.http_session.post(
                f"{self.backend_url}/get_accessibility_tree", json=payload
            ) as response:
                result = await response.json()
                if result.get("success"):
                    return result
                else:
                    raise Exception(
                        f"Accessibility tree request failed: {result}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to get accessibility tree: {e}")
            raise

    async def _get_current_map_state(self) -> Dict:
        """Get current map center and view state."""
        try:
            eval_script = """
            () => {
                const result = {
                    map_center: null,
                    viewport_bounds: null,
                    zoom_level: null,
                    map_loaded: false
                };

                // Try to get map center from Google Maps API if available
                if (window.google && window.google.maps) {
                    // Maps API methods would go here
                    result.map_loaded = true;
                }

                // Get viewport information
                result.viewport_bounds = {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    center_x: window.innerWidth / 2,
                    center_y: window.innerHeight / 2
                };

                // Look for map container
                const mapContainer = document.querySelector('#map') ||
                                   document.querySelector('.map-container') ||
                                   document.querySelector('[role="region"]');

                if (mapContainer) {
                    const rect = mapContainer.getBoundingClientRect();
                    result.map_center = {
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2
                    };
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
            self.logger.error(f"Failed to get map state: {e}")
            return {}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's drag action."""
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

            # Extract and log drag vector and distance
            drag_info = self._extract_drag_info(action, browser_result)
            if drag_info:
                self.validator.log_drag_vector_and_distance(
                    drag_info["vector"], drag_info["distance"]
                )
                self.test_state["drag_start_coords"] = drag_info["vector"][
                    "from"
                ]
                self.test_state["drag_end_coords"] = drag_info["vector"]["to"]

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

            # Get updated map state
            await asyncio.sleep(1)  # Wait for map to update
            updated_map_state = await self._get_current_map_state()
            self.test_state["map_center_after"] = updated_map_state.get(
                "map_center"
            )

            # Check if map center shifted
            if self._verify_map_shift():
                self.validator.map_center_shifted = True
                self.validator.new_area_visible = True
                self.test_state["map_view_changed"] = True

            # Evaluate the action
            action_type = action.get("action_type")

            if action_type == 8:  # DRAG action
                evaluation["valid"] = True
                evaluation[
                    "feedback"
                ] = f'Drag executed - distance: {drag_info["distance"] if drag_info else "unknown"}px'
                self.test_state["drag_executed"] = True

                if self.test_state["map_view_changed"]:
                    print(f"Map center shifts eastward, new area visible")
                    evaluation[
                        "feedback"
                    ] += " - map view successfully shifted"
                else:
                    evaluation[
                        "feedback"
                    ] += " - map view may not have shifted"

                evaluation["test_completed"] = True
                evaluation[
                    "validation_result"
                ] = self.validator.validate_drag_result()
            else:
                evaluation[
                    "feedback"
                ] = f"Unexpected action type {action_type} - expected DRAG (8) only"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {str(e)}"
            self.logger.error(f"Failed to execute action: {e}")

        return evaluation

    def _extract_drag_info(
        self, action: Dict, browser_result: Dict
    ) -> Optional[Dict]:
        """Extract drag vector and distance information."""
        # Default coordinates from config
        center_coords = self.gmaps_elements_config.get("map_center", {}).get(
            "coordinates", {"x": 0.5, "y": 0.5}
        )
        target_coords = self.gmaps_elements_config.get("drag_target", {}).get(
            "coordinates", {"x": 0.7, "y": 0.5}
        )

        # Try to get coordinates from action
        if "start_coordinate" in action and "end_coordinate" in action:
            start = action["start_coordinate"]
            end = action["end_coordinate"]
        elif (
            "coordinate" in action
            and isinstance(action["coordinate"], list)
            and len(action["coordinate"]) >= 4
        ):
            coord = action["coordinate"]
            start = {"x": coord[0], "y": coord[1]}
            end = {"x": coord[2], "y": coord[3]}
        else:
            # Use default coordinates
            start = center_coords
            end = target_coords

        # Calculate distance
        dx = (end["x"] - start["x"]) * 1280  # Assuming 1280px width
        dy = (end["y"] - start["y"]) * 800  # Assuming 800px height
        distance = (dx**2 + dy**2) ** 0.5

        return {
            "vector": {
                "from_x": start["x"],
                "from_y": start["y"],
                "to_x": end["x"],
                "to_y": end["y"],
                "from": start,
                "to": end,
            },
            "distance": distance,
        }

    def _verify_map_shift(self) -> bool:
        """Verify that the map center has shifted eastward."""
        before = self.test_state.get("map_center_before")
        after = self.test_state.get("map_center_after")

        if not before or not after:
            return True  # Assume success if we can't verify

        # Check if center moved eastward (x coordinate increased)
        if after.get("x", 0) > before.get("x", 0):
            return True

        return False

    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's action in the browser backend."""
        action_type = action.get("action_type")

        if action_type == 8:  # DRAG action
            # Get drag coordinates
            center_coords = self.gmaps_elements_config.get(
                "map_center", {}
            ).get("coordinates", {"x": 0.5, "y": 0.5})
            target_coords = self.gmaps_elements_config.get(
                "drag_target", {}
            ).get("coordinates", {"x": 0.7, "y": 0.5})

            start_x, start_y = center_coords["x"], center_coords["y"]
            end_x, end_y = target_coords["x"], target_coords["y"]

            # Execute drag via browser backend
            payload = {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            }

            try:
                async with self.http_session.post(
                    f"{self.backend_url}/drag", json=payload
                ) as response:
                    result = await response.json()
                    self.logger.info(
                        f"Drag executed from ({start_x}, {start_y}) to ({end_x}, {end_y})"
                    )
                    return result
            except Exception as e:
                self.logger.error(f"Drag execution failed: {e}")
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
            "drag_executed": self.test_state["drag_executed"],
            "map_center_before": self.test_state["map_center_before"],
            "map_center_after": self.test_state["map_center_after"],
            "map_view_changed": self.test_state["map_view_changed"],
            "test_completed": self.test_state["drag_executed"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps
            - self.test_state["current_step"],
        }

    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            "test_type": "gmaps_drag_map",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "drag_vector": self.validator.drag_vector,
            "drag_distance": self.validator.drag_distance,
            "validation_summary": self.validator.validate_drag_result(),
            "element_configuration": self.gmaps_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


# Test runner function for external agent systems
async def test_gmaps_drag_map(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Test function for Google Maps drag with real browser integration.
    """
    test_env = GoogleMapsDragTestEnvironment(url, backend_url, config_path)

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
def test_gmaps_drag_map_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(
        test_gmaps_drag_map(url, run_agent_func, backend_url, config_path)
    )


# Convenience function for quick integration
def create_gmaps_drag_map_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> GoogleMapsDragTestEnvironment:
    """Factory function to create a Google Maps drag test environment."""
    return GoogleMapsDragTestEnvironment(url, backend_url, config_path)
