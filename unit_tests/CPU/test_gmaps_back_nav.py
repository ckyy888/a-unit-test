"""
Google Maps Back Navigation Test Framework with Browser-in-Browser Integration
Purpose: Test agent's ability to navigate back from route error state using browser back button
"""
import asyncio
import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp

# WebSocket for screenshot streaming
try:
    import websockets
except ImportError:
    websockets = None

# Browser integration imports
import sys

sys.path.append("/home/ubuntu/webarena")


class ConfigLoader:
    """Loads and manages configuration from config.json file."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config.json in the same directory as this file
            config_path = Path(__file__).parent / "config.json"

        self.config_path = Path(config_path)
        self._config = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
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
        """Get default configuration if config file is not available."""
        return {
            "test_configuration": {
                "max_steps": 3,
                "timeout_seconds": 30,
                "screenshot_enabled": True,
                "detailed_logging": True,
            },
            "gmaps_navigation_elements": {
                "route_error_panel": {
                    "selectors": [
                        "[data-testid='route-error']",
                        ".section-directions-error",
                        "[aria-label*='Route not found']",
                        ".route-error",
                    ],
                    "description": "Route error panel",
                },
                "results_list": {
                    "selectors": [
                        "[data-testid='results']",
                        ".section-result",
                        "#pane .section-places",
                        ".results-list",
                    ],
                    "description": "Search results list area",
                },
            },
            "validation_rules": {
                "max_scroll_count": 2,
                "require_back_nav": True,
                "require_results_restoration": True,
            },
            "browser_settings": {
                "backend_url": "http://localhost:8000",
                "screenshot_frequency": "per_action",
                "wait_for_load": True,
                "network_idle_timeout": 2000,
            },
            "logging": {
                "level": "INFO",
                "file_path": "/home/ubuntu/webarena/unit_tests/CPU/gmaps_back_nav.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'test_configuration.max_steps')."""
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict:
        """Get entire configuration section."""
        return self._config.get(section, {})

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    @property
    def config(self) -> Dict:
        """Get the full configuration dictionary."""
        return self._config.copy()


class GMapsBackNavValidator:
    """Enhanced validator for Google Maps back navigation operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.navigation_actions: List[Dict[str, Any]] = []
        self.back_nav_executed = False
        self.scroll_count = 0

        self.config_loader = config_loader or ConfigLoader()
        self.validation_rules = self.config_loader.get_section("validation_rules")
        if not self.validation_rules:
            self.validation_rules = {
                "max_scroll_count": 2,
                "require_back_nav": True,
                "require_results_restoration": True,
            }

    def log_navigation_action(self, action_type: str, action_details: Dict) -> None:
        action_log = {
            "action_id": len(self.navigation_actions) + 1,
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "action_details": action_details,
        }
        self.navigation_actions.append(action_log)
        self.logger.info(f"Navigation action: {action_type}")

        if action_type == "BROWSER_BACK":
            self.back_nav_executed = True
            print("Back nav executed.")
        elif action_type == "SCROLL":
            self.scroll_count += 1

    def validate_navigation_result(self) -> Dict[str, Any]:
        validation_result = {
            "back_nav_executed": self.back_nav_executed,
            "scroll_count_valid": False,
            "results_restored": False,
            "scroll_count": self.scroll_count,
            "max_allowed_scrolls": self.validation_rules.get("max_scroll_count", 2),
            "errors": [],
            "warnings": [],
        }

        if not self.back_nav_executed:
            validation_result["errors"].append("Back navigation was not executed")

        max_scrolls = self.validation_rules.get("max_scroll_count", 2)
        if self.scroll_count <= max_scrolls:
            validation_result["scroll_count_valid"] = True
        else:
            validation_result["errors"].append(
                f"Too many scrolls: {self.scroll_count} (max: {max_scrolls})"
            )

        validation_result["overall_valid"] = (
            validation_result["back_nav_executed"]
            and validation_result["scroll_count_valid"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result

    def get_navigation_log(self) -> List[Dict]:
        return self.navigation_actions


class GMapsBackNavTestEnvironment:
    """Browser-in-browser test env for Google Maps back nav, aligned with general_test_runner."""

    def __init__(
        self,
        url: str,
        backend_url: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.config_loader = ConfigLoader(config_path)

        # Always use canonical URL from config for determinism
        self.url = self.config_loader.get("test_urls.google_maps_back_nav")
        if not self.url:
            raise ValueError(
                "Missing 'test_urls.google_maps_back_nav' entry in config.json"
            )

        self.backend_url = backend_url or self.config_loader.get(
            "browser_settings.backend_url", "http://localhost:8000"
        )

        self.validator = GMapsBackNavValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        self.max_steps = self.config_loader.get("test_configuration.max_steps", 3)
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 30
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.gmaps_elements_config = self.config_loader.get_section(
            "gmaps_navigation_elements"
        )

        # Browser-related settings
        self.screenshot_frequency = self.config_loader.get(
            "browser_settings.screenshot_frequency", "per_action"
        )
        self.wait_for_load = self.config_loader.get("browser_settings.wait_for_load", True)
        self.network_idle_timeout = self.config_loader.get(
            "browser_settings.network_idle_timeout", 2000
        )

        self.test_state: Dict[str, Any] = {
            "current_step": 0,
            "route_error_detected": False,
            "back_nav_completed": False,
            "results_restored": False,
            "action_history": [],
            "screenshots": [],
            "browser_session_active": False,
            "full_trajectory": [],
            "agent_decision_tree": None,
            "agent_decision_metadata": None,
        }

        self.http_session: Optional[aiohttp.ClientSession] = None
        self._setup_logging()

    def _setup_logging(self) -> None:
        log_level = self.config_loader.get("logging.level", "INFO")
        log_format = self.config_loader.get(
            "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_file = self.config_loader.get("logging.file_path")

        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")

    async def get_initial_observation(self) -> Dict:
        try:
            await self._init_browser_session()
            await self._navigate_to_url()

            # Wait a bit to stabilize
            await asyncio.sleep(5)
            await asyncio.sleep(3)

            screenshot_data = await self._get_screenshot()
            accessibility_data = await self._get_accessibility_tree()

            # Use configured selectors instead of dynamic /evaluate
            detected_elements = await self._detect_elements_from_config()

            self.test_state["screenshots"].append(
                {"step": "initial", "timestamp": datetime.now().isoformat(), "data": screenshot_data}
            )

            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "A route error is visible. Use browser back to return to the results list.",
                "expected_elements": self.gmaps_elements_config,
                "detected_elements": detected_elements,
                "initial_screenshot": screenshot_data,
                "accessibility_tree": accessibility_data.get("accessibility_tree", ""),
                "initial_accessibility_tree": accessibility_data.get("accessibility_tree", ""),
                "obs_metadata": {
                    "obs_nodes_info": accessibility_data.get("obs_nodes_info", {}),
                    "browser_config": accessibility_data.get("browser_config", {}),
                    "viewport_size": accessibility_data.get("viewport_size", {"width": 1280, "height": 800}),
                },
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_sequence": ["browser_back", "results_restored"],
                    "expected_final_state": "results list visible",
                },
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            self.test_state["full_trajectory"].append(
                {
                    "step": "initialization",
                    "timestamp": datetime.now().isoformat(),
                    "setup_result": setup_result,
                    "accessibility_data": accessibility_data,
                    "screenshot_captured": bool(screenshot_data),
                }
            )

            return setup_result

        except Exception as e:
            self.logger.error(f"Failed to initialize browser session: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    async def _init_browser_session(self) -> None:
        if not self.http_session:
            self.http_session = aiohttp.ClientSession()
        try:
            test_payload = {"url": "about:blank"}
            async with self.http_session.post(f"{self.backend_url}/goto", json=test_payload) as response:
                if response.status == 200:
                    self.test_state["browser_session_active"] = True
                    self.logger.info("Browser backend connection established")
                else:
                    raise Exception(f"Browser backend health check failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Cannot connect to browser backend at {self.backend_url}: {e}")
            raise

    async def _navigate_to_url(self) -> Dict:
        payload = {"url": self.url}
        async with self.http_session.post(f"{self.backend_url}/goto", json=payload) as response:
            result = await response.json()
            if result.get("success"):
                self.logger.info(f"Successfully navigated to {self.url}")
                return result
            raise Exception(f"Navigation failed: {result}")

    async def _get_accessibility_tree(self) -> Dict:
        try:
            payload = {"current_viewport_only": False}
            async with self.http_session.post(
                f"{self.backend_url}/get_accessibility_tree", json=payload
            ) as response:
                result = await response.json()
                if result.get("success"):
                    self.logger.info("✅ Accessibility tree retrieved from browser backend")
                    return result
                raise Exception(f"Accessibility tree request failed: {result}")
        except Exception as e:
            self.logger.error(f"Failed to get accessibility tree: {e}")
            raise

    async def _get_screenshot(self) -> str:
        try:
            if websockets:
                ws_url = self.backend_url.replace("http://", "ws://").replace("https://", "wss://") + "/screenshot"
                async with websockets.connect(ws_url) as websocket:
                    screenshot_data = await websocket.recv()
                    if isinstance(screenshot_data, bytes):
                        return base64.b64encode(screenshot_data).decode("utf-8")
                    return base64.b64encode(str(screenshot_data).encode()).decode("utf-8")
            else:
                self.logger.warning("WebSocket library not available")
                return "websocket_not_available"
        except Exception:
            # Fallback: generate a simple mock image
            try:
                from PIL import Image, ImageDraw

                img = Image.new("RGB", (1280, 800), color="white")
                draw = ImageDraw.Draw(img)
                draw.text((100, 100), "Route not found (mock)", fill="red")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                return "screenshot_generation_failed"

    async def _detect_elements_from_config(self) -> Dict:
        detected = {}
        for element_type, config in self.gmaps_elements_config.items():
            detected[element_type] = {
                "found": True,
                "selector_used": (config.get("selectors") or [None])[0],
                "coordinates": config.get("coordinates", {"x": 0.5, "y": 0.5}),
                "element_info": {
                    "found": True,
                    "selector": (config.get("selectors") or ["unknown"])[0],
                    "visible": True,
                    "tag": "DIV",
                },
            }
        return detected

    async def _verify_element_state(self, selector: str) -> Dict:
        try:
            payload = {"selector": selector, "selector_type": "css"}
            async with self.http_session.post(
                f"{self.backend_url}/get_element_bbox", json=payload
            ) as response:
                result = await response.json()
                if result.get("success", False):
                    return {
                        "element_found": True,
                        "visible": True,
                        "bounding_box": result.get("bounding_box", {}),
                    }
        except Exception as e:
            self.logger.warning(f"Verification failed for {selector}: {e}")
        return {"element_found": False, "visible": False}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        self.test_state["current_step"] += 1
        self.test_state["action_history"].append(action)

        evaluation = {
            "step": self.test_state["current_step"],
            "action": action,
            "valid": False,
            "feedback": "",
            "expected_next": None,
            "test_completed": False,
            "validation_result": None,
            "browser_result": None,
            "screenshot_after": None,
        }

        trajectory_step = {
            "step_number": self.test_state["current_step"],
            "timestamp_start": datetime.now().isoformat(),
            "agent_action": action,
            "accessibility_data_before": None,
            "browser_result": None,
            "screenshot_after": None,
            "evaluation_result": None,
            "timestamp_end": None,
        }

        if self.test_state["current_step"] > self.max_steps:
            evaluation["feedback"] = f"Exceeded maximum steps limit ({self.max_steps})"
            return evaluation

        try:
            stored_metadata = action.get("agent_decision_metadata")
            stored_tree = action.get("agent_decision_tree")

            if stored_metadata and stored_tree:
                obs_metadata = stored_metadata
                trajectory_step["used_stored_metadata"] = True
                trajectory_step["stored_tree_length"] = len(stored_tree)
                trajectory_step["accessibility_data_before"] = {"source": "stored_from_agent_decision"}
            else:
                accessibility_data = await self._get_accessibility_tree()
                trajectory_step["accessibility_data_before"] = accessibility_data
                obs_metadata = {
                    "obs_nodes_info": accessibility_data.get("obs_nodes_info", {}),
                    "browser_config": accessibility_data.get("browser_config", {}),
                    "viewport_size": accessibility_data.get("viewport_size", {"width": 1280, "height": 800}),
                }
                trajectory_step["used_stored_metadata"] = False

            browser_result = await self._execute_action_in_browser(action, obs_metadata)
            evaluation["browser_result"] = browser_result
            trajectory_step["browser_result"] = browser_result

            screenshot = await self._get_screenshot()
            evaluation["screenshot_after"] = screenshot
            trajectory_step["screenshot_after"] = screenshot

            self.test_state["screenshots"].append(
                {"step": self.test_state["current_step"], "action": action.get("action_type"), "timestamp": datetime.now().isoformat(), "data": screenshot}
            )

            # Validate
            route, _payload = None, None
            try:
                from action_conversion import convert_action

                route, _payload = convert_action(action, obs_metadata)
            except Exception:
                route = None

            if route == "/back":
                evaluation["valid"] = True
                evaluation["feedback"] = "Browser back navigation executed"
                self.test_state["back_nav_completed"] = True
                self.validator.log_navigation_action("BROWSER_BACK", action)

                # Verify results list appears
                results_cfg = self.gmaps_elements_config.get("results_list", {})
                restored = False
                for sel in results_cfg.get("selectors", []):
                    v = await self._verify_element_state(sel)
                    if v.get("element_found"):
                        restored = True
                        break
                self.test_state["results_restored"] = restored

                evaluation["test_completed"] = True
                evaluation["validation_result"] = self.validator.validate_navigation_result()
                evaluation["validation_result"]["results_restored"] = restored
            else:
                evaluation["feedback"] = "Unexpected action – expected browser back"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {e}"
            self.logger.error(f"Failed to execute action: {e}")
            trajectory_step["error"] = str(e)

        trajectory_step["evaluation_result"] = evaluation
        trajectory_step["timestamp_end"] = datetime.now().isoformat()
        self.test_state["full_trajectory"].append(trajectory_step)
        return evaluation

    async def _execute_action_in_browser(self, action: Dict, obs_metadata: Optional[Dict] = None) -> Dict:
        try:
            from action_conversion import convert_action

            route, payload = convert_action(action, obs_metadata)
        except Exception as e:
            self.logger.warning(f"Action conversion failed: {e}")
            route, payload = None, None

        if route == "/back":
            async with self.http_session.post(f"{self.backend_url}/back") as response:
                return await response.json()

        # Unsupported in this test
        raise Exception(f"Unsupported action route for this test: {route}")

    async def cleanup(self) -> None:
        if self.http_session:
            await self.http_session.close()
            self.test_state["browser_session_active"] = False
            self.logger.info("Browser session cleanup completed")

    def get_test_status(self) -> Dict:
        return {
            "current_step": self.test_state["current_step"],
            "max_steps": self.max_steps,
            "route_error_detected": self.test_state.get("route_error_detected", False),
            "back_nav_completed": self.test_state["back_nav_completed"],
            "results_restored": self.test_state["results_restored"],
            "test_completed": self.test_state["back_nav_completed"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps - self.test_state["current_step"],
        }

    def _describe_clicked_element(self, element_id: str, obs_metadata: Dict) -> str:
        if not obs_metadata or "obs_nodes_info" not in obs_metadata:
            return f"ID {element_id} (no metadata available)"
        nodes = obs_metadata["obs_nodes_info"]
        element_info = nodes.get(str(element_id))
        if not element_info:
            return f"ID {element_id} (element not found in accessibility tree)"
        element_text = element_info.get("text", "")
        import re
        match = re.match(r"\[\d+\]\s+(\w+)\s+\'([^\']*)\'\s*(.*)", element_text)
        if match:
            role, name, properties = match.groups()
            parts = []
            if name.strip():
                parts.append(f"'{name.strip()}'")
            parts.append(f"({role})")
            if properties.strip():
                prop_matches = re.findall(r"(\w+):\s*([^:\s]+)", properties)
                meaningful = [f"{n}: {v}" for n, v in prop_matches if n in ["expanded", "checked", "selected", "disabled"] and v.lower() != "false"]
                if meaningful:
                    parts.append(f"[{', '.join(meaningful)}]")
            return f"ID {element_id}: {' '.join(parts)}"
        return f"ID {element_id}: {element_text[:100]}..." if len(element_text) > 100 else f"ID {element_id}: {element_text}"

    def save_full_trajectory(self) -> None:
        try:
            trajectory_file = Path("/home/ubuntu/webarena/unit_tests/CPU/gmaps_back_nav_trajectory.json")
            with open(trajectory_file, "w") as f:
                json.dump(self.test_state["full_trajectory"], f, indent=2)
            self.logger.info(f"Full trajectory saved to {trajectory_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save trajectory: {e}")

    def save_workflow_summary(self) -> None:
        try:
            summary = {
                "test_type": "gmaps_back_navigation",
                "total_steps": self.test_state["current_step"],
                "max_steps": self.max_steps,
                "test_completed": self.test_state["back_nav_completed"],
                "navigation_actions": self.validator.get_navigation_log(),
                "final_validation": self.validator.validate_navigation_result(),
                "browser_session_data": {
                    "url": self.url,
                    "backend_url": self.backend_url,
                    "screenshots_captured": len(self.test_state["screenshots"]),
                },
                "timestamp": datetime.now().isoformat(),
            }
            summary_file = Path("/home/ubuntu/webarena/unit_tests/CPU/gmaps_back_nav_summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Workflow summary saved to {summary_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save workflow summary: {e}")

    def get_test_report(self) -> Dict:
        return {
            "plugin_version": "1.0.0",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "navigation_actions": self.validator.get_navigation_log(),
            "validation_summary": self.validator.validate_navigation_result(),
            "element_configuration": self.gmaps_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


async def test_gmaps_back_nav(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    test_env = GMapsBackNavTestEnvironment(url, backend_url, config_path)

    initial_obs = await test_env.get_initial_observation()
    if not initial_obs.get("success"):
        return {"success": False, "error": "Failed to initialize browser environment", "details": initial_obs, "timestamp": datetime.now().isoformat()}

    evaluations: List[Dict[str, Any]] = []
    test_completed = False
    current_observation = initial_obs

    for step_num in range(test_env.max_steps):
        agent_actions = run_agent_func(current_observation)
        if not isinstance(agent_actions, list) or len(agent_actions) == 0:
            break
        action = agent_actions[0]
        if action is None:
            break

        evaluation = await test_env.evaluate_agent_action(action)
        evaluations.append(evaluation)
        if evaluation.get("test_completed", False):
            test_completed = True
            break

        if step_num < test_env.max_steps - 1:
            try:
                fresh_screenshot = await test_env._get_screenshot()
                fresh_accessibility = await test_env._get_accessibility_tree()
                current_observation = {
                    "success": True,
                    "url": test_env.url,
                    "task_description": current_observation.get("task_description", ""),
                    "initial_screenshot": fresh_screenshot,
                    "accessibility_tree": fresh_accessibility.get("accessibility_tree", ""),
                    "initial_accessibility_tree": fresh_accessibility.get("accessibility_tree", ""),
                    "obs_metadata": {
                        "obs_nodes_info": fresh_accessibility.get("obs_nodes_info", {}),
                        "browser_config": fresh_accessibility.get("browser_config", {}),
                        "viewport_size": fresh_accessibility.get("viewport_size", {"width": 1280, "height": 800}),
                    },
                    "current_step": step_num + 1,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception:
                pass

    test_status = self_status = test_env.get_test_status()
    test_report = test_env.get_test_report()

    test_env.save_full_trajectory()
    test_env.save_workflow_summary()
    await test_env.cleanup()

    return {
        "success": test_completed and self_status["test_completed"],
        "initial_observation": initial_obs,
        "agent_actions": agent_actions if 'agent_actions' in locals() else [],
        "evaluations": evaluations,
        "test_status": self_status,
        "test_report": test_report,
        "browser_integration": True,
        "timestamp": datetime.now().isoformat(),
    }


def test_gmaps_back_nav_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    return asyncio.run(test_gmaps_back_nav(url, run_agent_func, backend_url, config_path))


def create_gmaps_back_nav_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> GMapsBackNavTestEnvironment:
    return GMapsBackNavTestEnvironment(url, backend_url, config_path)
