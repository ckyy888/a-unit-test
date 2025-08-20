"""
BBC Article Pagination Test Framework
Purpose: Test agent's ability to navigate to next page of multi-page article
"""
import asyncio
import base64
import json
import logging
import re

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


class BBCPaginationValidator:
    """Validator for BBC article pagination operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.page_transitions = []
        self.scrolled_down = False
        self.next_page_clicked = False
        self.page_2_content_loaded = False
        self.url_parameter_correct = False

        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()

    def log_page_transition(
        self, transition_type: str, page_data: Dict
    ) -> None:
        """Log page transitions."""
        transition_log = {
            "transition_id": len(self.page_transitions) + 1,
            "timestamp": datetime.now().isoformat(),
            "transition_type": transition_type,
            "page_data": page_data,
        }

        self.page_transitions.append(transition_log)
        self.logger.info(f"Page transition: {transition_type}")

        if transition_type == "SCROLL_DOWN":
            self.scrolled_down = True
            print(
                f"Page transitions: Scrolled down to access pagination controls"
            )
        elif transition_type == "NEXT_PAGE_CLICKED":
            self.next_page_clicked = True
            print(
                f"Page transitions: Next page button clicked, navigating to page {page_data.get('target_page', '2')}"
            )
        elif transition_type == "PAGE_2_CONTENT_LOADED":
            self.page_2_content_loaded = True
            print(f"Page transitions: Page 2 content successfully loaded")
        elif transition_type == "URL_PARAMETER_VERIFIED":
            self.url_parameter_correct = True
            print(
                f"Page transitions: URL contains correct page parameter - {page_data.get('url_parameter', 'page=2')}"
            )

    def validate_pagination_result(self) -> Dict[str, Any]:
        """Validate the complete pagination sequence."""
        validation_result = {
            "scrolled_down": self.scrolled_down,
            "next_page_clicked": self.next_page_clicked,
            "page_2_content_loaded": self.page_2_content_loaded,
            "url_parameter_correct": self.url_parameter_correct,
            "pagination_successful": False,
            "errors": [],
            "warnings": [],
        }

        # Check if scroll down was performed
        if not self.scrolled_down:
            validation_result["errors"].append(
                "Did not scroll down to access pagination controls"
            )

        # Check if next page was clicked
        if not self.next_page_clicked:
            validation_result["errors"].append(
                "Next page button was not clicked"
            )

        # Check if page 2 content loaded
        if not self.page_2_content_loaded:
            validation_result["errors"].append("Page 2 content did not load")

        # Check if URL parameter is correct
        if not self.url_parameter_correct:
            validation_result["warnings"].append(
                "URL does not contain correct page parameter"
            )

        # Determine if pagination was successful
        validation_result["pagination_successful"] = (
            self.scrolled_down
            and self.next_page_clicked
            and self.page_2_content_loaded
        )

        # Overall validation status
        validation_result["overall_valid"] = (
            validation_result["pagination_successful"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result

    def get_transition_log(self) -> List[Dict]:
        """Return the complete log of page transitions."""
        return self.page_transitions


class BBCPaginationTestEnvironment:
    """Test environment for BBC article pagination with browser-in-browser integration."""

    def __init__(
        self,
        url: str,
        backend_url: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)

        # Use canonical URL from config
        self.url = self.config_loader.get("test_urls.bbc_article_pagination") or url
        if not self.url:
            raise ValueError("Missing 'test_urls.bbc_article_pagination' in config.json")
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get(
            "browser_settings.backend_url", "http://localhost:8000"
        )

        # Initialize validator with config
        self.validator = BBCPaginationValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 30
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.bbc_elements_config = self.config_loader.get_section(
            "bbc_article_elements"
        )

        self.test_state = {
            "current_step": 0,
            "current_page": 1,
            "total_pages": 3,
            "scrolled_to_bottom": False,
            "next_page_visible": False,
            "page_content_changed": False,
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
        """Get initial test setup with page 1 of 3-page article."""
        try:
            # Initialize browser session
            await self._init_browser_session()

            # Navigate to the URL (should show page 1 of article)
            await self._navigate_to_url()

            # Take initial screenshot and accessibility tree
            screenshot_data = await self._get_screenshot()
            acc = await self._get_accessibility_tree()

            # Detect pagination elements
            element_detection = await self._detect_pagination_elements()
            page_info = await self._get_current_page_info()

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": "initial_page_1",
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot_data,
                }
            )

            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "Navigate to page 2 of multi-page article using pagination controls",
                "element_detection": element_detection,
                "page_info": page_info,
                "initial_screenshot": screenshot_data,
                "accessibility_tree": acc.get("accessibility_tree", ""),
                "initial_accessibility_tree": acc.get("accessibility_tree", ""),
                "obs_metadata": {
                    "obs_nodes_info": acc.get("obs_nodes_info", {}),
                    "browser_config": acc.get("browser_config", {}),
                    "viewport_size": acc.get("viewport_size", {"width": 1280, "height": 800}),
                },
                "pagination_elements": self.bbc_elements_config,
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_sequence": [
                        "SCROLL_DOWN",
                        "CLICK(next_page)",
                        "verify content change",
                    ],
                    "expected_final_state": "page 2 content loaded, URL contains page parameter",
                },
                "history": [
                    "BBC News multi-page article loaded",
                    "Currently on page 1 of 3-page article",
                    "Article content is displayed",
                    "Pagination controls are at bottom of page",
                    "Need to scroll down to access next page button",
                ],
                "current_article": {
                    "page": 1,
                    "total_pages": 3,
                    "next_page_available": True,
                },
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Update test state based on detection
            if element_detection.get("next_page", {}).get("found"):
                self.test_state["next_page_visible"] = True

            self.logger.info(
                f"BBC article pagination test initialized: {self.url}"
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
                d.text((100, 100), "BBC pagination (mock)", fill="black")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                return "screenshot_generation_failed"

    async def _detect_pagination_elements(self) -> Dict:
        """Detect pagination elements on the current page."""
        detected_elements = {}

        for element_type, config in self.bbc_elements_config.items():
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
                                in_viewport: rect.top >= 0 && rect.top <= window.innerHeight,
                                text_content: element.textContent || '',
                                href: element.href || '',
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

    async def _get_current_page_info(self) -> Dict:
        """Get current page information and URL."""
        try:
            eval_script = """
            () => {
                const result = {
                    current_url: window.location.href,
                    page_parameter: null,
                    article_content_length: 0,
                    scroll_position: window.pageYOffset,
                    page_height: document.body.scrollHeight,
                    viewport_height: window.innerHeight
                };

                // Check for page parameter in URL
                const urlParams = new URLSearchParams(window.location.search);
                result.page_parameter = urlParams.get('page') || urlParams.get('p');

                // Get article content length
                const articleContent = document.querySelector('.article-body') ||
                                     document.querySelector('[data-testid="article-content"]') ||
                                     document.querySelector('.story-content');

                if (articleContent) {
                    result.article_content_length = articleContent.textContent.length;
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
            self.logger.error(f"Failed to get page info: {e}")
            return {}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's pagination action."""
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
            # Execute the action in browser
            browser_result = await self._execute_action_in_browser(action)
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

            # Get updated page state
            page_info = await self._get_current_page_info()

            # Evaluate the action
            action_type = action.get("action_type")
            element_name = action.get("element_name", "").lower()

            if action_type == 7:  # SCROLL_DOWN action
                evaluation["valid"] = True
                evaluation[
                    "feedback"
                ] = "Scrolled down to access pagination controls"
                self.test_state["scrolled_to_bottom"] = True

                # Log the scroll action
                self.validator.log_page_transition(
                    "SCROLL_DOWN",
                    {
                        "scroll_position": page_info.get("scroll_position", 0),
                        "page_height": page_info.get("page_height", 0),
                    },
                )

                evaluation["expected_next"] = "Click next page button"

            elif action_type == 6:  # CLICK action
                if "next" in element_name or "page" in element_name:
                    # Next page button clicked
                    evaluation["valid"] = True
                    evaluation[
                        "feedback"
                    ] = "Next page button clicked successfully"
                    self.test_state["current_page"] = 2

                    # Log the page transition
                    self.validator.log_page_transition(
                        "NEXT_PAGE_CLICKED", {"target_page": 2, "from_page": 1}
                    )

                    # Check if page content changed and URL updated
                    await asyncio.sleep(2)  # Wait for page load
                    updated_page_info = await self._get_current_page_info()

                    # Verify content change
                    content_changed = await self._verify_content_change(
                        page_info, updated_page_info
                    )
                    if content_changed:
                        self.test_state["page_content_changed"] = True

                        # Log successful content load
                        self.validator.log_page_transition(
                            "PAGE_2_CONTENT_LOADED",
                            {
                                "new_content_length": updated_page_info.get(
                                    "article_content_length", 0
                                ),
                                "page_2_loaded": True,
                            },
                        )

                        # Check URL parameter
                        if self._check_url_parameter(updated_page_info):
                            self.validator.log_page_transition(
                                "URL_PARAMETER_VERIFIED",
                                {
                                    "url_parameter": updated_page_info.get(
                                        "page_parameter"
                                    )
                                    or "page=2",
                                    "url": updated_page_info.get(
                                        "current_url", ""
                                    ),
                                },
                            )

                        evaluation["test_completed"] = True
                        evaluation[
                            "validation_result"
                        ] = self.validator.validate_pagination_result()
                    else:
                        evaluation[
                            "feedback"
                        ] = "Next page clicked but content did not change"
                else:
                    evaluation[
                        "feedback"
                    ] = f"Clicked wrong element - expected next page button, got: {element_name}"
            else:
                evaluation[
                    "feedback"
                ] = f"Unexpected action type {action_type} - expected SCROLL_DOWN (7) or CLICK (6)"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {str(e)}"
            self.logger.error(f"Failed to execute action: {e}")

        return evaluation

    async def _execute_action_in_browser(self, action: Dict, obs_metadata: Optional[Dict] = None) -> Dict:
        """Execute the agent's action in the browser backend."""
        action_type = action.get("action_type")

        if action_type == 6:  # CLICK action
            # Prefer convert_action
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
                    if "next" in element_name or "page" in element_name:
                        coords = self.bbc_elements_config.get("next_page", {}).get("coordinates", {})
                        x, y = coords.get("x", 0.8), coords.get("y", 0.9)
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

        elif action_type == 7:  # SCROLL_DOWN action
            # Execute scroll down via browser backend
            payload = {"dx": 0, "dy": 500}  # Scroll down 500 pixels

            try:
                async with self.http_session.post(
                    f"{self.backend_url}/scroll", json=payload
                ) as response:
                    result = await response.json()
                    self.logger.info(f"Scroll down executed: {result}")
                    return result
            except Exception as e:
                self.logger.error(f"Scroll execution failed: {e}")
                raise
        else:
            raise Exception(
                f"Unsupported action type for this test: {action_type}"
            )

    async def _verify_content_change(
        self, before_info: Dict, after_info: Dict
    ) -> bool:
        """Verify that page content actually changed."""
        before_length = before_info.get("article_content_length", 0)
        after_length = after_info.get("article_content_length", 0)

        # Content should change (different article section)
        if (
            abs(before_length - after_length) > 100
        ):  # Significant content change
            return True

        # Check URL change
        before_url = before_info.get("current_url", "")
        after_url = after_info.get("current_url", "")

        if before_url != after_url:
            return True

        return False

    def _check_url_parameter(self, page_info: Dict) -> bool:
        """Check if URL contains correct page parameter."""
        url = page_info.get("current_url", "")
        page_param = page_info.get("page_parameter")

        # Check for page=2 or similar parameter
        if page_param and (page_param == "2" or page_param == 2):
            return True

        # Check URL for page indicators
        if re.search(r"[?&]page=2|[?&]p=2|/page/2|/2/", url):
            return True

        return False

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
            "current_page": self.test_state["current_page"],
            "scrolled_to_bottom": self.test_state["scrolled_to_bottom"],
            "next_page_visible": self.test_state["next_page_visible"],
            "page_content_changed": self.test_state["page_content_changed"],
            "test_completed": self.test_state["page_content_changed"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps
            - self.test_state["current_step"],
        }

    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            "test_type": "bbc_article_pagination",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "page_transitions": self.validator.get_transition_log(),
            "validation_summary": self.validator.validate_pagination_result(),
            "element_configuration": self.bbc_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


# Test runner function for external agent systems
async def test_bbc_article_pagination(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Test function for BBC article pagination with real browser integration.
    """
    test_env = BBCPaginationTestEnvironment(url, backend_url, config_path)

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
def test_bbc_article_pagination_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(
        test_bbc_article_pagination(
            url, run_agent_func, backend_url, config_path
        )
    )


# Convenience function for quick integration
def create_bbc_article_pagination_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> BBCPaginationTestEnvironment:
    """Factory function to create a BBC article pagination test environment."""
    return BBCPaginationTestEnvironment(url, backend_url, config_path)
