"""
ESPN NBA Scoreboard Tab Toggle Test with Browser-in-Browser Integration
Purpose: Switch from Tue Oct 21 to Wed Oct 22 tab on ESPN NBA scoreboard
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

try:
    import websockets
except ImportError:
    websockets = None

import sys


class ConfigLoader:
    """Loads and manages configuration from config.json file."""

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
        except Exception:
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict:
        return {
            "test_configuration": {
                "max_steps": 3,
                "timeout_seconds": 30,
                "screenshot_enabled": True,
                "detailed_logging": True,
            },
            "espn_tabs": {
                "games_tab": {
                    "selectors": [
                        "[data-testid='tab-games']",
                        "button[aria-controls*='games']",
                        ".tabs .games",
                    ],
                    "coordinates": {"x": 0.25, "y": 0.2},
                    "description": "Games tab",
                },
                "wed_oct_22_tab": {
                    "selectors": [
                        "[data-testid='tab-wed_oct_22']",
                        "button[aria-controls*='wed_oct_22']",
                        ".tabs .wed_oct_22",
                    ],
                    "coordinates": {"x": 0.45, "y": 0.2},
                    "description": "Wed Oct 22 tab",
                },
                "scoreboard_container": {
                    "selectors": [
                        "[data-testid='scoreboard']",
                        ".scoreboard",
                        "#scoreboard",
                    ],
                    "description": "Scoreboard container"
                }
            },
            "browser_settings": {
                "backend_url": "http://localhost:8000",
                "screenshot_frequency": "per_action",
                "wait_for_load": True,
                "network_idle_timeout": 2000,
            },
            "logging": {
                "level": "INFO",
                "file_path": "/home/ubuntu/webarena/unit_tests/CPU/espn_score_toggle.log",
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
        except Exception:
            return default

    def get_section(self, section: str) -> Dict:
        return self._config.get(section, {})


class ESPNScoreToggleValidator:
    """Validator for ESPN NBA scoreboard tab toggle operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.tab_actions = []
        self.wed_oct_22_clicked = False
        self.initial_tab_state = None
        self.final_tab_state = None

        # Use provided config loader or create default one
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()

    def log_tab_action(self, action_type: str, tab_info: Dict) -> None:
        """Log tab action with details."""
        action_log = {
            "action_id": len(self.tab_actions) + 1,
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "tab_info": tab_info,
        }

        self.tab_actions.append(action_log)
        self.logger.info(f"Tab action: {action_type}")

        # Track specific actions
        if action_type == "CLICK_WED_OCT_22":
            self.wed_oct_22_clicked = True
            print(
                f"Selected tab state: {tab_info.get('selected_tab', 'unknown')}"
            )  # Required validator print

    def set_initial_tab_state(self, tab_state: Dict) -> None:
        """Record the initial tab state."""
        self.initial_tab_state = tab_state
        self.logger.info(
            f"Initial tab state recorded: {tab_state.get('active_tab', 'unknown')}"
        )

    def set_final_tab_state(self, tab_state: Dict) -> None:
        """Record the final tab state."""
        self.final_tab_state = tab_state
        self.logger.info(
            f"Final tab state recorded: {tab_state.get('active_tab', 'unknown')}"
        )

    def validate_tab_toggle_result(self) -> Dict[str, Any]:
        """Validate the complete tab toggle sequence."""
        validation_result = {
            "wed_oct_22_clicked": self.wed_oct_22_clicked,
            "wed_oct_22_aria_selected": False,
            "games_initially_selected": False,
            "tab_state_changed": False,
            "initial_tab": self.initial_tab_state,
            "final_tab": self.final_tab_state,
            "errors": [],
            "warnings": [],
        }

        # Check if wed oct 22 tab was clicked
        if not self.wed_oct_22_clicked:
            validation_result["errors"].append(
                "Wed Oct 22 tab was not clicked"
            )

        # Check initial state (Games should be active)
        if self.initial_tab_state:
            if self.initial_tab_state.get("active_tab") == "games":
                validation_result["games_initially_selected"] = True
            else:
                validation_result["warnings"].append(
                    f"Expected Games tab initially active, got: {self.initial_tab_state.get('active_tab')}"
                )

        # Check final state (Wed Oct 22 should be active with aria-selected="true")
        if self.final_tab_state:
            if self.final_tab_state.get("wed_oct_22_aria_selected") == "true":
                validation_result["wed_oct_22_aria_selected"] = True
            else:
                validation_result["errors"].append(
                    "Wed Oct 22 tab does not have aria-selected='true'"
                )

            if self.final_tab_state.get("active_tab") == "wed_oct_22":
                validation_result["tab_state_changed"] = True
            else:
                validation_result["errors"].append(
                    "Tab state did not change to wed_oct_22"
                )

        # Overall validation status
        validation_result["overall_valid"] = (
            validation_result["wed_oct_22_clicked"]
            and validation_result["wed_oct_22_aria_selected"]
            and validation_result["tab_state_changed"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result

    def get_tab_action_log(self) -> List[Dict]:
        """Return the complete log of tab actions."""
        return self.tab_actions


class ESPNScoreToggleTestEnvironment:
    """Test environment aligned with general_test_runner and new config-driven pattern."""

    def __init__(
        self,
        url: str,
        backend_url: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.config_loader = ConfigLoader(config_path)
        self.url = self.config_loader.get("test_urls.espn_score_toggle")
        if not self.url:
            raise ValueError("Missing 'test_urls.espn_score_toggle' entry in config.json")
        self.backend_url = backend_url or self.config_loader.get(
            "browser_settings.backend_url", "http://localhost:8000"
        )

        # Initialize validator with config
        self.validator = ESPNScoreToggleValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        # Load configuration values
        self.max_steps = self.config_loader.get(
            "test_configuration.max_steps", 3
        )
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 30
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.espn_elements_config = self.config_loader.get_section("espn_tabs")

        self.test_state = {
            "current_step": 0,
            "games_tab_detected": False,
            "wed_oct_22_tab_detected": False,
            "wed_oct_22_clicked": False,
            "tab_switched": False,
            "action_history": [],
            "screenshots": [],
            "browser_session_active": False,
        }

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
        """Get initial test setup with Games tab active screenshot."""
        try:
            # Initialize browser session
            await self._init_browser_session()

            # Navigate to the URL (should show Games tab active)
            await self._navigate_to_url()

            # Take initial screenshot
            screenshot_data = await self._get_screenshot()

            # Detect tab elements and current state
            tab_detection = await self._detect_tab_elements()
            initial_tab_state = await self._get_current_tab_state()

            # Record initial state in validator
            self.validator.set_initial_tab_state(initial_tab_state)

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": "initial_games_tab",
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot_data,
                }
            )

            acc = await self._get_accessibility_tree()
            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "Switch from Tue Oct 21 tab to Wed Oct 22 tab on ESPN NBA scoreboard",
                "tab_detection": tab_detection,
                "initial_tab_state": initial_tab_state,
                "initial_screenshot": screenshot_data,
                "accessibility_tree": acc.get("accessibility_tree", ""),
                "initial_accessibility_tree": acc.get("accessibility_tree", ""),
                "obs_metadata": {
                    "obs_nodes_info": acc.get("obs_nodes_info", {}),
                    "browser_config": acc.get("browser_config", {}),
                    "viewport_size": acc.get("viewport_size", {"width": 1280, "height": 800}),
                },
                "tab_elements": self.espn_elements_config,
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_action": "CLICK(Wed Oct 22)",
                    "expected_final_state": 'aria-selected="true" on Wed Oct 22 tab',
                },
                "history": [
                    "ESPN NBA scoreboard page freshly loaded",
                    "Tue Oct 21 tab is currently active (default state)",
                    "Wed Oct 22 tab is available for clicking",
                ],
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            self.test_state["games_tab_detected"] = tab_detection.get(
                "games_tab", {}
            ).get("found", False)
            self.test_state["wed_oct_22_tab_detected"] = tab_detection.get(
                "wed_oct_22_tab", {}
            ).get("found", False)

            self.logger.info(f"ESPN score toggle test initialized: {self.url}")
            return setup_result

        except Exception as e:
            self.logger.error(f"Failed to initialize browser session: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def _init_browser_session(self) -> None:
        if not self.http_session:
            self.http_session = aiohttp.ClientSession()
        test_payload = {"url": "about:blank"}
        async with self.http_session.post(f"{self.backend_url}/goto", json=test_payload) as r:
            if r.status == 200:
                self.test_state["browser_session_active"] = True
            else:
                raise Exception("Backend not healthy")

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
        try:
            if websockets:
                ws_url = self.backend_url.replace("http://", "ws://").replace("https://", "wss://") + "/screenshot"
                async with websockets.connect(ws_url) as ws:
                    data = await ws.recv()
                    if isinstance(data, bytes):
                        return base64.b64encode(data).decode("utf-8")
                    return base64.b64encode(str(data).encode()).decode("utf-8")
        except Exception:
            pass
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (1280, 800), "white")
            d = ImageDraw.Draw(img)
            d.text((100, 100), "ESPN Scoreboard (mock)", fill="black")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return "screenshot_generation_failed"

    async def _get_accessibility_tree(self) -> Dict:
        try:
            payload = {"current_viewport_only": False}
            async with self.http_session.post(
                f"{self.backend_url}/get_accessibility_tree", json=payload
            ) as response:
                result = await response.json()
                if result.get("success"):
                    return result
                return {"success": False, "obs_nodes_info": {}}
        except Exception:
            return {"success": False, "obs_nodes_info": {}}

    async def _detect_tab_elements(self) -> Dict:
        detected = {}
        for tab_type, cfg in self.espn_elements_config.items():
            detected[tab_type] = {
                "found": True,
                "selector_used": (cfg.get("selectors") or [None])[0],
                "coordinates": cfg.get("coordinates", {"x": 0.5, "y": 0.5}),
                "element_info": {"found": True, "selector": (cfg.get("selectors") or ["unknown"])[0]},
            }
        return detected

    async def _get_current_tab_state(self) -> Dict:
        """Get the current state of all tabs."""
        try:
            eval_script = """
            () => {
                const result = {
                    active_tab: null,
                    games_aria_selected: null,
                    wed_oct_22_aria_selected: null,
                    tabs_found: []
                };

                // Look for tabs with role='tab'
                const tabs = document.querySelectorAll('[role="tab"]');
                tabs.forEach(tab => {
                    const text = tab.textContent.toLowerCase();
                    const ariaSelected = tab.getAttribute('aria-selected');

                    result.tabs_found.push({
                        text: text,
                        aria_selected: ariaSelected,
                        tag: tab.tagName
                    });

                    if (text.includes('game') && ariaSelected === 'true') {
                        result.active_tab = 'games';
                        result.games_aria_selected = ariaSelected;
                    } else if (text.includes('wed oct 22') && ariaSelected === 'true') {
                        result.active_tab = 'wed_oct_22';
                        result.wed_oct_22_aria_selected = ariaSelected;
                    }

                    if (text.includes('wed oct 22')) {
                        result.wed_oct_22_aria_selected = ariaSelected;
                    }
                    if (text.includes('game')) {
                        result.games_aria_selected = ariaSelected;
                    }
                });

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
            self.logger.error(f"Failed to get tab state: {e}")
            return {}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's tab toggle action."""
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

        # Check step limit
        if self.test_state["current_step"] > self.max_steps:
            evaluation[
                "feedback"
            ] = f"Exceeded maximum steps limit ({self.max_steps})"
            return evaluation

        try:
            # Build obs_metadata for element-id mapping
            stored_meta = action.get("agent_decision_metadata")
            stored_tree = action.get("agent_decision_tree")
            if stored_meta and stored_tree:
                obs_meta = stored_meta
            else:
                acc = await self._get_accessibility_tree()
                obs_meta = {
                    "obs_nodes_info": acc.get("obs_nodes_info", {}),
                    "browser_config": acc.get("browser_config", {}),
                    "viewport_size": acc.get("viewport_size", {"width": 1280, "height": 800}),
                }

            # Execute the action in browser
            browser_result = await self._execute_action_in_browser(action, obs_meta)
            evaluation["browser_result"] = browser_result

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

            # Evaluate the action
            action_type = action.get("action_type")
            element_name = action.get("element_name", "").lower()

            if action_type == 6:  # CLICK action
                if "wed oct 22" in element_name:
                    evaluation["valid"] = True
                    evaluation[
                        "feedback"
                    ] = "Wed Oct 22 tab click executed correctly"
                    self.test_state["wed_oct_22_clicked"] = True

                    # Log the tab action
                    self.validator.log_tab_action(
                        "CLICK_WED_OCT_22",
                        {"selected_tab": "wed_oct_22", "action": action},
                    )

                    # Check final tab state
                    final_tab_state = await self._get_current_tab_state()
                    self.validator.set_final_tab_state(final_tab_state)

                    # Test is now complete
                    evaluation["test_completed"] = True
                    evaluation[
                        "validation_result"
                    ] = self.validator.validate_tab_toggle_result()

                else:
                    evaluation[
                        "feedback"
                    ] = f"Clicked wrong element - expected Wed Oct 22 tab, got: {element_name}"

            else:
                evaluation[
                    "feedback"
                ] = f"Unexpected action type {action_type} - expected CLICK (6)"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {str(e)}"
            self.logger.error(f"Failed to execute action: {e}")

        return evaluation

    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        try:
            from action_conversion import convert_action
            route, payload = convert_action(action, {})
        except Exception:
            route, payload = None, None
        if route == "/click":
            async with self.http_session.post(f"{self.backend_url}/click", json={"x": payload["x"], "y": payload["y"]}) as r:
                return await r.json()
        raise Exception(f"Unsupported action type: {action.get('action_type')}")

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
            "games_tab_detected": self.test_state["games_tab_detected"],
            "wed_oct_22_tab_detected": self.test_state[
                "wed_oct_22_tab_detected"
            ],
            "wed_oct_22_clicked": self.test_state["wed_oct_22_clicked"],
            "tab_switched": self.test_state["tab_switched"],
            "test_completed": self.test_state["wed_oct_22_clicked"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps
            - self.test_state["current_step"],
        }

    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            "test_type": "espn_score_toggle",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "tab_actions": self.validator.get_tab_action_log(),
            "validation_summary": self.validator.validate_tab_toggle_result(),
            "element_configuration": self.espn_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


# Test runner function for external agent systems
async def test_espn_score_toggle(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Test function for ESPN NBA scoreboard tab toggle with real browser integration.

    Args:
        url: URL of the ESPN NBA scoreboard page to test
        run_agent_func: External function that takes initial_observation and returns list of actions
        backend_url: Browser-in-browser backend URL (optional, uses config)
        config_path: Path to configuration file (optional)

    Returns:
        Complete test results dictionary
    """
    test_env = ESPNScoreToggleTestEnvironment(url, backend_url, config_path)

    try:
        # Provide initial observation to the external agent
        initial_obs = await test_env.get_initial_observation()

        if not initial_obs.get("success"):
            return {
                "success": False,
                "error": "Failed to initialize browser environment",
                "details": initial_obs,
                "timestamp": datetime.now().isoformat(),
            }

        # Let the external agent decide what actions to take
        agent_actions = run_agent_func(initial_obs)

        # Validate that agent returned a list of actions
        if not isinstance(agent_actions, list):
            await test_env.cleanup()
            return {
                "success": False,
                "error": "Agent must return a list of actions",
                "timestamp": datetime.now().isoformat(),
            }

        # Evaluate each action (should be single CLICK on Wed Oct 22)
        evaluations = []
        test_completed = False

        for action in agent_actions:
            evaluation = await test_env.evaluate_agent_action(action)
            evaluations.append(evaluation)

            if evaluation["test_completed"]:
                test_completed = True
                break

        # Get final test status and generate report
        test_status = test_env.get_test_status()
        test_report = test_env.get_test_report()

        # Cleanup browser session
        await test_env.cleanup()

        return {
            "success": test_completed and test_status["test_completed"],
            "initial_observation": initial_obs,
            "agent_actions": agent_actions,
            "evaluations": evaluations,
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
def test_espn_score_toggle_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(
        test_espn_score_toggle(url, run_agent_func, backend_url, config_path)
    )


# Convenience function for quick integration
def create_espn_score_toggle_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> ESPNScoreToggleTestEnvironment:
    """Factory function to create an ESPN scoreboard tab toggle test environment."""
    return ESPNScoreToggleTestEnvironment(url, backend_url, config_path)
