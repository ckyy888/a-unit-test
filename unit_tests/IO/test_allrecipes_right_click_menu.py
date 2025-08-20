"""
AllRecipes Right Click Menu Test Framework
Purpose: Test agent's ability to perform right click to open context menu
Tests: SINGLE CLICK (RIGHT) action only
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
from browser_env.actions import ActionTypes, create_right_click_action

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


class AllRecipesRightClickValidator:
    """Validator for AllRecipes right click operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.right_click_executed = False
        self.right_click_coordinates = None
        self.context_menu_appeared = False
        self.save_image_option_found = False

        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()

    def log_right_click_coordinates(self, coordinates: Dict) -> None:
        """Print right-click coordinates."""
        self.right_click_coordinates = coordinates
        self.right_click_executed = True
        print(
            f"Right-click coordinates: ({coordinates['x']}, {coordinates['y']})"
        )
        self.logger.info(f"Right-click executed at coordinates: {coordinates}")

    def validate_right_click_result(self) -> Dict[str, Any]:
        """Validate the right click action result."""
        validation_result = {
            "right_click_executed": self.right_click_executed,
            "coordinates_logged": self.right_click_coordinates is not None,
            "context_menu_appeared": self.context_menu_appeared,
            "save_image_option_found": self.save_image_option_found,
            "right_click_successful": False,
            "errors": [],
            "warnings": [],
        }

        # Check if right click was executed
        if not self.right_click_executed:
            validation_result["errors"].append(
                "Right click action was not executed"
            )

        # Check if coordinates were logged
        if not self.right_click_coordinates:
            validation_result["errors"].append(
                "Right-click coordinates were not logged"
            )

        # Check if context menu appeared
        if not self.context_menu_appeared:
            validation_result["errors"].append("Context menu did not appear")

        # Check if Save image option is found
        if not self.save_image_option_found:
            validation_result["warnings"].append(
                "'Save image' option not found in context menu"
            )

        # Determine if right click was successful
        validation_result["right_click_successful"] = (
            self.right_click_executed and self.context_menu_appeared
        )

        # Overall validation status
        validation_result["overall_valid"] = (
            validation_result["right_click_successful"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result


class AllRecipesRightClickTestEnvironment:
    """Test environment for AllRecipes right click with browser-in-browser integration."""

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
        self.validator = AllRecipesRightClickValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        # Load configuration values
        self.max_steps = 1  # Single action test (right-click only)
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 10
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.allrecipes_elements_config = self.config_loader.get_section(
            "allrecipes_right_click_elements"
        )

        self.test_state = {
            "current_step": 0,
            "right_click_executed": False,
            "context_menu_visible_before": False,
            "context_menu_visible_after": False,
            "save_image_option_present": False,
            "recipe_image_found": False,
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
        """Get initial test setup with recipe image ready for right-click."""
        try:
            # Initialize browser session
            await self._init_browser_session()

            # Navigate to the URL (should show AllRecipes recipe grid)
            await self._navigate_to_url()

            # Take initial screenshot
            screenshot_data = await self._get_screenshot()

            # Detect recipe image and context menu elements
            element_detection = await self._detect_recipe_elements()
            menu_state = await self._get_current_menu_state()

            # Store initial state
            self.test_state["recipe_image_found"] = element_detection.get(
                "recipe_image", {}
            ).get("found", False)
            self.test_state["context_menu_visible_before"] = menu_state.get(
                "context_menu_visible", False
            )

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": "initial_recipe_grid",
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot_data,
                }
            )

            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "Execute right click on recipe image to open context menu",
                "element_detection": element_detection,
                "menu_state": menu_state,
                "initial_screenshot": screenshot_data,
                "right_click_elements": self.allrecipes_elements_config,
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_action": "RIGHT_CLICK(recipe_image) once only",
                    "expected_result": 'context menu appears with "Save image" option',
                },
                "history": [
                    "AllRecipes recipe grid page loaded",
                    "Recipe cards with images are displayed",
                    "Recipe images are visible and ready for interaction",
                    "Context menu should appear on right-click",
                    "Ready for right-click action",
                ],
                "current_recipe_grid": {
                    "recipe_image_found": self.test_state[
                        "recipe_image_found"
                    ],
                    "context_menu_visible": self.test_state[
                        "context_menu_visible_before"
                    ],
                    "right_click_target_ready": True,
                },
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"AllRecipes right click test initialized: {self.url}"
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

    async def _detect_recipe_elements(self) -> Dict:
        """Detect recipe image and context menu elements."""
        detected_elements = {}

        for element_type, config in self.allrecipes_elements_config.items():
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
                                src: element.src || '',
                                alt: element.alt || '',
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

    async def _get_current_menu_state(self) -> Dict:
        """Get current context menu state."""
        try:
            eval_script = """
            () => {
                const result = {
                    context_menu_visible: false,
                    save_image_option_present: false,
                    menu_items_count: 0
                };

                // Check for context menu
                const contextMenus = [
                    document.querySelector('.context-menu'),
                    document.querySelector('[role="menu"]'),
                    document.querySelector('.right-click-menu'),
                    document.querySelector('.contextmenu')
                ];

                for (const menu of contextMenus) {
                    if (menu) {
                        const rect = menu.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            result.context_menu_visible = true;

                            // Count menu items
                            const menuItems = menu.querySelectorAll('[role="menuitem"], .menu-item, li');
                            result.menu_items_count = menuItems.length;

                            // Check for Save image option
                            for (const item of menuItems) {
                                const text = item.textContent || item.innerText || '';
                                if (text.toLowerCase().includes('save') && text.toLowerCase().includes('image')) {
                                    result.save_image_option_present = true;
                                    break;
                                }
                            }
                            break;
                        }
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
            self.logger.error(f"Failed to get menu state: {e}")
            return {}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's right click action."""
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

            # Log right-click coordinates
            if "coordinates" in action:
                self.validator.log_right_click_coordinates(
                    action["coordinates"]
                )
            elif browser_result and "click_coordinates" in browser_result:
                self.validator.log_right_click_coordinates(
                    browser_result["click_coordinates"]
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

            # Get updated menu state
            await asyncio.sleep(0.5)  # Wait for context menu to appear
            updated_menu_state = await self._get_current_menu_state()
            self.test_state[
                "context_menu_visible_after"
            ] = updated_menu_state.get("context_menu_visible", False)
            self.test_state[
                "save_image_option_present"
            ] = updated_menu_state.get("save_image_option_present", False)

            # Check if context menu appeared
            if self.test_state["context_menu_visible_after"]:
                self.validator.context_menu_appeared = True
                print(f"Context menu appears with save image option")

                # Check for Save image option
                if self.test_state["save_image_option_present"]:
                    self.validator.save_image_option_found = True

            # Evaluate the action
            action_type = action.get("action_type")
            element_name = action.get("element_name", "").lower()

            if action_type == 12:  # RIGHT_CLICK action
                if "recipe" in element_name and "image" in element_name:
                    evaluation["valid"] = True
                    evaluation[
                        "feedback"
                    ] = "Right-click executed on recipe image"
                    self.test_state["right_click_executed"] = True

                    if self.test_state["context_menu_visible_after"]:
                        evaluation[
                            "feedback"
                        ] += " - context menu successfully appeared"
                        if self.test_state["save_image_option_present"]:
                            evaluation[
                                "feedback"
                            ] += ' with "Save image" option'
                    else:
                        evaluation[
                            "feedback"
                        ] += " - context menu did not appear"

                    evaluation["test_completed"] = True
                    evaluation[
                        "validation_result"
                    ] = self.validator.validate_right_click_result()
                else:
                    evaluation[
                        "feedback"
                    ] = f"Right-clicked wrong element - expected recipe image, got: {element_name}"
            else:
                evaluation[
                    "feedback"
                ] = f"Unexpected action type {action_type} - expected RIGHT_CLICK (12) only"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {str(e)}"
            self.logger.error(f"Failed to execute action: {e}")

        return evaluation

    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's right-click action in the browser backend."""
        action_type = action.get("action_type")

        if action_type == 12:  # RIGHT_CLICK action
            # Determine right-click coordinates
            if "coordinates" in action:
                x, y = action["coordinates"]["x"], action["coordinates"]["y"]
            else:
                coords = self.allrecipes_elements_config.get(
                    "recipe_image", {}
                ).get("coordinates", {"x": 0.3, "y": 0.6})
                x, y = coords["x"], coords["y"]

            # Execute right-click via browser backend
            payload = {
                "x": x,
                "y": y,
                "button": "right",  # Specify right mouse button
            }

            try:
                async with self.http_session.post(
                    f"{self.backend_url}/click", json=payload
                ) as response:
                    result = await response.json()
                    result["click_coordinates"] = {"x": x, "y": y}
                    result["button"] = "right"
                    self.logger.info(f"Right-click executed at ({x}, {y})")
                    return result
            except Exception as e:
                self.logger.error(f"Right-click execution failed: {e}")
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
            "right_click_executed": self.test_state["right_click_executed"],
            "recipe_image_found": self.test_state["recipe_image_found"],
            "context_menu_visible_before": self.test_state[
                "context_menu_visible_before"
            ],
            "context_menu_visible_after": self.test_state[
                "context_menu_visible_after"
            ],
            "save_image_option_present": self.test_state[
                "save_image_option_present"
            ],
            "test_completed": self.test_state["right_click_executed"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps
            - self.test_state["current_step"],
        }

    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            "test_type": "allrecipes_right_click_menu",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "right_click_coordinates": self.validator.right_click_coordinates,
            "validation_summary": self.validator.validate_right_click_result(),
            "element_configuration": self.allrecipes_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


# Test runner function for external agent systems
async def test_allrecipes_right_click_menu(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Test function for AllRecipes right click menu with real browser integration.
    """
    test_env = AllRecipesRightClickTestEnvironment(
        url, backend_url, config_path
    )

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
def test_allrecipes_right_click_menu_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(
        test_allrecipes_right_click_menu(
            url, run_agent_func, backend_url, config_path
        )
    )


# Convenience function for quick integration
def create_allrecipes_right_click_menu_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> AllRecipesRightClickTestEnvironment:
    """Factory function to create an AllRecipes right click menu test environment."""
    return AllRecipesRightClickTestEnvironment(url, backend_url, config_path)
