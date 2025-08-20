"""
Amazon Product Variant Selection Logic Test Framework
Purpose: Test agent's ability to select valid size/color combination to enable "Add to Cart"
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
from browser_env.actions import ActionTypes, create_click_action

# Import base configuration loader
sys.path.append("/home/ubuntu/webarena/unit_tests/CPU")


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


class AmazonVariantValidator:
    """Validator for Amazon product variant selection operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.variant_selections = []
        self.size_selected = False
        self.color_selected = False
        self.add_to_cart_enabled = False
        self.price_updated = False

        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()

    def log_variant_selection(
        self, selection_type: str, variant_data: Dict
    ) -> None:
        """Log variant selections."""
        selection_log = {
            "selection_id": len(self.variant_selections) + 1,
            "timestamp": datetime.now().isoformat(),
            "selection_type": selection_type,
            "variant_data": variant_data,
        }

        self.variant_selections.append(selection_log)
        self.logger.info(f"Variant selection: {selection_type}")

        if selection_type == "SIZE_13_INCH":
            self.size_selected = True
            print(
                f"Variant selections: Size changed to {variant_data.get('size', '13 inch')}"
            )
        elif selection_type == "COLOR_AVAILABLE":
            self.color_selected = True
            print(
                f"Variant selections: Color changed to {variant_data.get('color', 'available option')}"
            )

        # Check if Add to Cart is enabled and price updated
        if variant_data.get("add_to_cart_enabled"):
            self.add_to_cart_enabled = True
        if variant_data.get("price_updated"):
            self.price_updated = True
            print(
                f"Variant selections: Price updated to {variant_data.get('new_price', 'updated price')}"
            )

    def validate_variant_selection_result(self) -> Dict[str, Any]:
        """Validate the complete variant selection sequence."""
        validation_result = {
            "size_selected": self.size_selected,
            "color_selected": self.color_selected,
            "add_to_cart_enabled": self.add_to_cart_enabled,
            "price_updated": self.price_updated,
            "valid_combination_found": False,
            "errors": [],
            "warnings": [],
        }

        # Check if both size and color were selected
        if not self.size_selected:
            validation_result["errors"].append("13-inch size was not selected")
        if not self.color_selected:
            validation_result["errors"].append(
                "Available color was not selected"
            )

        # Check if Add to Cart became enabled
        if not self.add_to_cart_enabled:
            validation_result["errors"].append(
                "Add to Cart button did not become enabled"
            )

        # Check if price was updated
        if not self.price_updated:
            validation_result["warnings"].append(
                "Price was not updated after variant selection"
            )

        # Determine if valid combination was found
        validation_result["valid_combination_found"] = (
            self.size_selected
            and self.color_selected
            and self.add_to_cart_enabled
        )

        # Overall validation status
        validation_result["overall_valid"] = (
            validation_result["valid_combination_found"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result

    def get_variant_log(self) -> List[Dict]:
        """Return the complete log of variant selections."""
        return self.variant_selections


class AmazonVariantTestEnvironment:
    """Test environment for Amazon variant selection logic with browser-in-browser integration."""

    def __init__(
        self,
        url: str,
        backend_url: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)

        # Use canonical URL from config
        self.url = self.config_loader.get("test_urls.amazon_variant_selection") or url
        if not self.url:
            raise ValueError("Missing 'test_urls.amazon_variant_selection' in config.json")
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get(
            "browser_settings.backend_url", "http://localhost:8000"
        )

        # Initialize validator with config
        self.validator = AmazonVariantValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 30
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.amazon_elements_config = self.config_loader.get_section(
            "amazon_variant_elements"
        )

        self.test_state = {
            "current_step": 0,
            "current_size": "15_inch",  # Starting with unavailable 15" model
            "current_color": None,
            "size_changed": False,
            "color_changed": False,
            "add_to_cart_enabled": False,
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
        """Get initial test setup with 15" model currently unavailable."""
        try:
            # Initialize browser session
            await self._init_browser_session()

            # Navigate to the URL (should show laptop with variant options)
            await self._navigate_to_url()

            # Take initial screenshot and accessibility tree
            screenshot_data = await self._get_screenshot()
            acc = await self._get_accessibility_tree()

            # Detect variant elements and availability
            element_detection = await self._detect_variant_elements()
            variant_state = await self._get_current_variant_state()

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": "initial_variant_unavailable",
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot_data,
                }
            )

            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "Select valid size and color combination to enable Add to Cart button",
                "element_detection": element_detection,
                "variant_state": variant_state,
                "initial_screenshot": screenshot_data,
                "accessibility_tree": acc.get("accessibility_tree", ""),
                "initial_accessibility_tree": acc.get("accessibility_tree", ""),
                "obs_metadata": {
                    "obs_nodes_info": acc.get("obs_nodes_info", {}),
                    "browser_config": acc.get("browser_config", {}),
                    "viewport_size": acc.get("viewport_size", {"width": 1280, "height": 800}),
                },
                "variant_elements": self.amazon_elements_config,
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_sequence": [
                        'CLICK(13" size)',
                        "CLICK(available_color)",
                    ],
                    "expected_final_state": "Add to Cart enabled, price updated",
                },
                "history": [
                    "Amazon laptop product page loaded",
                    "Currently viewing 15-inch model",
                    "15-inch model is currently unavailable",
                    "Multiple size and color variants available",
                    "Some size/color combinations are out of stock",
                    "Need to find valid combination",
                ],
                "current_variant": {
                    "size": "15_inch",
                    "available": False,
                    "add_to_cart_enabled": False,
                },
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Update test state based on detection
            if element_detection.get("size_13_inch", {}).get("found"):
                self.test_state["size_options_available"] = True
            if element_detection.get("available_color", {}).get("found"):
                self.test_state["color_options_available"] = True

            self.logger.info(
                f"Amazon variant selection test initialized: {self.url}"
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
        """Try WebSocket screenshot first; fallback to HTTP or mock image."""
        try:
            try:
                import websockets  # type: ignore
                ws_url = self.backend_url.replace("http://", "ws://").replace("https://", "wss://") + "/screenshot"
                async with websockets.connect(ws_url) as ws:
                    data = await ws.recv()
                    if isinstance(data, bytes):
                        return base64.b64encode(data).decode("utf-8")
                    return base64.b64encode(str(data).encode()).decode("utf-8")
            except Exception:
                pass

            async with self.http_session.get(f"{self.backend_url}/screenshot") as response:
                if response.status == 200:
                    screenshot_bytes = await response.read()
                    return base64.b64encode(screenshot_bytes).decode("utf-8")
                raise Exception(f"Screenshot request failed: {response.status}")
        except Exception:
            try:
                from PIL import Image, ImageDraw
                img = Image.new("RGB", (1280, 800), "white")
                d = ImageDraw.Draw(img)
                d.text((100, 100), "Amazon variant (mock)", fill="black")
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

    async def _detect_variant_elements(self) -> Dict:
        """Detect variant selector elements on the current page."""
        detected_elements = {}

        for element_type, config in self.amazon_elements_config.items():
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
                                enabled: !element.disabled,
                                available: !element.classList.contains('unavailable'),
                                text_content: element.textContent || '',
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

    async def _get_current_variant_state(self) -> Dict:
        """Get the current variant selection state."""
        try:
            eval_script = """
            () => {
                const result = {
                    current_size: null,
                    current_color: null,
                    add_to_cart_enabled: false,
                    price: null,
                    availability: 'unknown'
                };

                // Check Add to Cart button state
                const addToCartButton = document.querySelector('#add-to-cart-button') ||
                                      document.querySelector('[data-testid="add-to-cart"]') ||
                                      document.querySelector('input[name="submit.add-to-cart"]');

                if (addToCartButton) {
                    result.add_to_cart_enabled = !addToCartButton.disabled &&
                                               !addToCartButton.classList.contains('disabled');
                }

                // Check current price
                const priceElement = document.querySelector('.a-price-whole') ||
                                   document.querySelector('.a-price') ||
                                   document.querySelector('[data-testid="price"]');

                if (priceElement) {
                    result.price = priceElement.textContent || '';
                }

                // Check availability message
                const availabilityElement = document.querySelector('#availability span') ||
                                          document.querySelector('.availability-message') ||
                                          document.querySelector('[data-testid="availability"]');

                if (availabilityElement) {
                    result.availability = availabilityElement.textContent || '';
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
            self.logger.error(f"Failed to get variant state: {e}")
            return {}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's variant selection action."""
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
                obs_metadata = stored_meta
            else:
                acc = await self._get_accessibility_tree()
                obs_metadata = {
                    "obs_nodes_info": acc.get("obs_nodes_info", {}),
                    "browser_config": acc.get("browser_config", {}),
                    "viewport_size": acc.get("viewport_size", {"width": 1280, "height": 800}),
                }

            # Execute the action in browser
            browser_result = await self._execute_action_in_browser(action, obs_metadata)
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

            # Get updated variant state
            variant_state = await self._get_current_variant_state()

            # Evaluate the action
            action_type = action.get("action_type")
            element_name = action.get("element_name", "").lower()

            if action_type == 6:  # CLICK action
                if "13" in element_name and "inch" in element_name:
                    # Size selection to 13-inch
                    evaluation["valid"] = True
                    evaluation[
                        "feedback"
                    ] = "13-inch size selected successfully"
                    self.test_state["current_size"] = "13_inch"
                    self.test_state["size_changed"] = True

                    # Log the variant selection
                    self.validator.log_variant_selection(
                        "SIZE_13_INCH",
                        {
                            "size": "13 inch",
                            "available": True,
                            "add_to_cart_enabled": variant_state.get(
                                "add_to_cart_enabled", False
                            ),
                            "price_updated": bool(variant_state.get("price")),
                        },
                    )

                    if not self.test_state["color_changed"]:
                        evaluation[
                            "expected_next"
                        ] = "Select an available color option"
                    else:
                        # Both size and color selected, check if Add to Cart is enabled
                        if variant_state.get("add_to_cart_enabled"):
                            evaluation["test_completed"] = True
                            evaluation[
                                "validation_result"
                            ] = (
                                self.validator.validate_variant_selection_result()
                            )

                elif "color" in element_name or "available" in element_name:
                    # Color selection
                    evaluation["valid"] = True
                    evaluation[
                        "feedback"
                    ] = "Available color selected successfully"
                    self.test_state["current_color"] = "available"
                    self.test_state["color_changed"] = True

                    # Log the variant selection
                    self.validator.log_variant_selection(
                        "COLOR_AVAILABLE",
                        {
                            "color": "available option",
                            "add_to_cart_enabled": variant_state.get(
                                "add_to_cart_enabled", False
                            ),
                            "price_updated": bool(variant_state.get("price")),
                            "new_price": variant_state.get("price", ""),
                        },
                    )

                    if not self.test_state["size_changed"]:
                        evaluation[
                            "expected_next"
                        ] = "Select 13-inch size option"
                    else:
                        # Both size and color selected, check if Add to Cart is enabled
                        if variant_state.get("add_to_cart_enabled"):
                            evaluation["test_completed"] = True
                            evaluation[
                                "validation_result"
                            ] = (
                                self.validator.validate_variant_selection_result()
                            )

                else:
                    evaluation[
                        "feedback"
                    ] = f"Clicked wrong element - expected size or color option, got: {element_name}"
            else:
                evaluation[
                    "feedback"
                ] = f"Unexpected action type {action_type} - expected CLICK (6)"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {str(e)}"
            self.logger.error(f"Failed to execute action: {e}")

        return evaluation

    async def _execute_action_in_browser(self, action: Dict, obs_metadata: Optional[Dict] = None) -> Dict:
        """Execute the agent's action in the browser backend."""
        action_type = action.get("action_type")

        if action_type == 6:  # CLICK action
            # Prefer convert_action for mapping
            try:
                from action_conversion import convert_action
                route, payload = convert_action(action, obs_metadata)
                if route == "/click":
                    x, y = payload["x"], payload["y"]
                else:
                    raise ValueError("Unsupported route")
            except Exception:
                if "coordinates" in action:
                    x, y = action["coordinates"]["x"], action["coordinates"]["y"]
                elif "element_name" in action:
                    element_name = action["element_name"].lower()
                    if "13" in element_name and "inch" in element_name:
                        coords = self.amazon_elements_config.get("size_13_inch", {}).get("coordinates", {})
                        x, y = coords.get("x", 0.3), coords.get("y", 0.6)
                    elif "color" in element_name or "available" in element_name:
                        coords = self.amazon_elements_config.get("available_color", {}).get("coordinates", {})
                        x, y = coords.get("x", 0.4), coords.get("y", 0.7)
                    else:
                        x, y = 0.5, 0.5
                else:
                    x, y = 0.5, 0.5

            # Execute click via browser backend
            payload = {"x": x, "y": y}

            try:
                async with self.http_session.post(
                    f"{self.backend_url}/click", json=payload
                ) as response:
                    result = await response.json()
                    self.logger.info(f"Click executed at ({x}, {y}): {result}")
                    return result
            except Exception as e:
                self.logger.error(f"Click execution failed: {e}")
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
            "current_size": self.test_state["current_size"],
            "current_color": self.test_state["current_color"],
            "size_changed": self.test_state["size_changed"],
            "color_changed": self.test_state["color_changed"],
            "add_to_cart_enabled": self.test_state["add_to_cart_enabled"],
            "test_completed": self.test_state["size_changed"]
            and self.test_state["color_changed"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps
            - self.test_state["current_step"],
        }

    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            "test_type": "amazon_variant_selection_logic",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "variant_selections": self.validator.get_variant_log(),
            "validation_summary": self.validator.validate_variant_selection_result(),
            "element_configuration": self.amazon_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


# Test runner function for external agent systems
async def test_amazon_variant_selection_logic(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Test function for Amazon variant selection logic with real browser integration.
    """
    test_env = AmazonVariantTestEnvironment(url, backend_url, config_path)

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

        evaluations = []
        test_completed = False

        for action in agent_actions:
            evaluation = await test_env.evaluate_agent_action(action)
            evaluations.append(evaluation)

            if evaluation["test_completed"]:
                test_completed = True
                break

        test_status = test_env.get_test_status()
        test_report = test_env.get_test_report()

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
def test_amazon_variant_selection_logic_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(
        test_amazon_variant_selection_logic(
            url, run_agent_func, backend_url, config_path
        )
    )


# Convenience function for quick integration
def create_amazon_variant_selection_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> AmazonVariantTestEnvironment:
    """Factory function to create an Amazon variant selection test environment."""
    return AmazonVariantTestEnvironment(url, backend_url, config_path)
