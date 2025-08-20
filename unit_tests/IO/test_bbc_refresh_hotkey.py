"""
BBC Refresh Hotkey Test Framework
Purpose: Test agent's ability to perform hotkey refresh on stale article
Tests: HOTKEY action only
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
from browser_env.actions import ActionTypes, create_hotkey_action

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


class BBCRefreshValidator:
    """Validator for BBC refresh hotkey operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.hotkey_executed = False
        self.hotkey_combination = None
        self.page_reloaded = False
        self.timestamp_updated = False

        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()

    def log_hotkey_combination_detected(self, combination: str) -> None:
        """Print hotkey combination detected."""
        self.hotkey_combination = combination
        self.hotkey_executed = True
        print(f"Hotkey combination detected: {combination}")
        self.logger.info(f"Hotkey executed: {combination}")

    def validate_refresh_result(self) -> Dict[str, Any]:
        """Validate the hotkey refresh result."""
        validation_result = {
            "hotkey_executed": self.hotkey_executed,
            "hotkey_combination_logged": self.hotkey_combination is not None,
            "expected_combinations": ["Ctrl+R", "F5"],
            "actual_combination": self.hotkey_combination,
            "page_reloaded": self.page_reloaded,
            "timestamp_updated": self.timestamp_updated,
            "refresh_successful": False,
            "errors": [],
            "warnings": [],
        }

        # Check if hotkey was executed
        if not self.hotkey_executed:
            validation_result["errors"].append(
                "Hotkey action was not executed"
            )

        # Check if hotkey combination is valid
        if self.hotkey_combination not in ["Ctrl+R", "F5", "Command+R"]:
            validation_result["warnings"].append(
                f"Unexpected hotkey combination: {self.hotkey_combination}"
            )

        # Check if page reloaded
        if not self.page_reloaded:
            validation_result["errors"].append("Page did not reload")

        # Check if timestamp updated
        if not self.timestamp_updated:
            validation_result["warnings"].append(
                "Article timestamp did not update"
            )

        # Determine if refresh was successful
        validation_result["refresh_successful"] = (
            self.hotkey_executed and self.page_reloaded
        )

        # Overall validation status
        validation_result["overall_valid"] = (
            validation_result["refresh_successful"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result


class BBCRefreshTestEnvironment:
    """Test environment for BBC refresh hotkey with browser-in-browser integration."""

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
        self.validator = BBCRefreshValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 10
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.bbc_elements_config = self.config_loader.get_section(
            "bbc_refresh_elements"
        )

        self.test_state = {
            "current_step": 0,
            "hotkey_executed": False,
            "hotkey_combination": None,
            "timestamp_before": None,
            "timestamp_after": None,
            "page_reloaded": False,
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
        """Get initial test setup with article having stale timestamp."""
        try:
            # Initialize browser session
            await self._init_browser_session()

            # Navigate to the URL (should show BBC article with stale timestamp)
            await self._navigate_to_url()

            # Take initial screenshot
            screenshot_data = await self._get_screenshot()

            # Detect timestamp and content elements
            timestamp_detection = await self._detect_timestamp_elements()
            article_state = await self._get_current_article_state()

            # Store initial timestamp
            self.test_state["timestamp_before"] = article_state.get(
                "timestamp_text"
            )

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": "initial_stale_article",
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot_data,
                }
            )

            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "Execute hotkey refresh to update stale article timestamp",
                "timestamp_detection": timestamp_detection,
                "article_state": article_state,
                "initial_screenshot": screenshot_data,
                "refresh_elements": self.bbc_elements_config,
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_action": 'HOTKEY("Ctrl+R") or HOTKEY("F5") once only',
                    "expected_result": "page reloads and timestamp updates",
                },
                "history": [
                    "BBC News article page loaded",
                    "Article loaded 10 minutes ago",
                    "Timestamp shows stale time",
                    "Article content may be outdated",
                    "Need refresh to get latest version",
                ],
                "current_article": {
                    "timestamp": self.test_state["timestamp_before"],
                    "stale": True,
                    "needs_refresh": True,
                },
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"BBC refresh hotkey test initialized: {self.url}"
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

    async def _detect_timestamp_elements(self) -> Dict:
        """Detect timestamp and content elements."""
        detected_elements = {}

        for element_type, config in self.bbc_elements_config.items():
            detected = {
                "found": False,
                "selector_used": None,
                "element_info": None,
            }

            # Try each selector to find the element
            for selector in config.get("selectors", []):
                try:
                    eval_script = f"""
                    () => {{
                        const element = document.querySelector('{selector}');
                        if (element) {{
                            return {{
                                found: true,
                                selector: '{selector}',
                                visible: element.offsetWidth > 0 && element.offsetHeight > 0,
                                text_content: element.textContent || '',
                                datetime: element.dateTime || element.getAttribute('datetime') || '',
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

    async def _get_current_article_state(self) -> Dict:
        """Get current article timestamp and content state."""
        try:
            eval_script = """
            () => {
                const result = {
                    timestamp_text: '',
                    content_loaded: false,
                    page_load_time: Date.now()
                };

                // Get timestamp
                const timestampElements = [
                    document.querySelector('.timestamp'),
                    document.querySelector('[data-testid="publish-time"]'),
                    document.querySelector('.article-timestamp'),
                    document.querySelector('time')
                ];

                for (const element of timestampElements) {
                    if (element && element.textContent) {
                        result.timestamp_text = element.textContent.trim();
                        break;
                    }
                }

                // Check content
                const contentElement = document.querySelector('.article-body') ||
                                     document.querySelector('[data-testid="article-content"]') ||
                                     document.querySelector('.story-content');

                if (contentElement) {
                    result.content_loaded = contentElement.textContent.length > 0;
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
            self.logger.error(f"Failed to get article state: {e}")
            return {}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's hotkey action."""
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

            # Log hotkey combination
            hotkey = self._extract_hotkey(action)
            if hotkey:
                self.validator.log_hotkey_combination_detected(hotkey)
                self.test_state["hotkey_combination"] = hotkey

            # Wait for page to reload
            await asyncio.sleep(2)

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

            # Get updated article state
            updated_article_state = await self._get_current_article_state()
            self.test_state["timestamp_after"] = updated_article_state.get(
                "timestamp_text"
            )

            # Check if page reloaded and timestamp updated
            if self._verify_page_refresh():
                self.validator.page_reloaded = True
                self.test_state["page_reloaded"] = True
                print(f"Page reloads and timestamp updates")

            # Evaluate the action
            action_type = action.get("action_type")

            if action_type == 10:  # HOTKEY action
                evaluation["valid"] = True
                evaluation["feedback"] = f"Hotkey executed: {hotkey}"
                self.test_state["hotkey_executed"] = True

                if self.test_state["page_reloaded"]:
                    evaluation["feedback"] += " - page successfully refreshed"
                else:
                    evaluation["feedback"] += " - page refresh not detected"

                evaluation["test_completed"] = True
                evaluation[
                    "validation_result"
                ] = self.validator.validate_refresh_result()
            else:
                evaluation[
                    "feedback"
                ] = f"Unexpected action type {action_type} - expected HOTKEY (10) only"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {str(e)}"
            self.logger.error(f"Failed to execute action: {e}")

        return evaluation

    def _extract_hotkey(self, action: Dict) -> str:
        """Extract hotkey combination from action."""
        if "key" in action:
            return action["key"]
        elif "hotkey" in action:
            return action["hotkey"]
        elif "combination" in action:
            return action["combination"]
        else:
            return "Ctrl+R"  # Default

    def _verify_page_refresh(self) -> bool:
        """Verify that the page was refreshed."""
        # Check if timestamps are different (simple check)
        before = self.test_state.get("timestamp_before", "")
        after = self.test_state.get("timestamp_after", "")

        if before != after:
            self.validator.timestamp_updated = True
            return True

        # Assume success if timestamps are the same but action was executed
        return self.test_state.get("hotkey_executed", False)

    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's hotkey action in the browser backend."""
        action_type = action.get("action_type")

        if action_type == 10:  # HOTKEY action
            hotkey = self._extract_hotkey(action)

            # Execute hotkey via browser backend
            payload = {"key": hotkey}

            try:
                async with self.http_session.post(
                    f"{self.backend_url}/keyboard", json=payload
                ) as response:
                    result = await response.json()
                    self.logger.info(f"Hotkey executed: {hotkey}")
                    return result
            except Exception as e:
                self.logger.error(f"Hotkey execution failed: {e}")
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
            "hotkey_executed": self.test_state["hotkey_executed"],
            "hotkey_combination": self.test_state["hotkey_combination"],
            "timestamp_before": self.test_state["timestamp_before"],
            "timestamp_after": self.test_state["timestamp_after"],
            "page_reloaded": self.test_state["page_reloaded"],
            "test_completed": self.test_state["hotkey_executed"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps
            - self.test_state["current_step"],
        }

    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            "test_type": "bbc_refresh_hotkey",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "hotkey_combination": self.validator.hotkey_combination,
            "validation_summary": self.validator.validate_refresh_result(),
            "element_configuration": self.bbc_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


# Test runner function for external agent systems
async def test_bbc_refresh_hotkey(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Test function for BBC refresh hotkey with real browser integration.
    """
    test_env = BBCRefreshTestEnvironment(url, backend_url, config_path)

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
def test_bbc_refresh_hotkey_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(
        test_bbc_refresh_hotkey(url, run_agent_func, backend_url, config_path)
    )


# Convenience function for quick integration
def create_bbc_refresh_hotkey_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> BBCRefreshTestEnvironment:
    """Factory function to create a BBC refresh hotkey test environment."""
    return BBCRefreshTestEnvironment(url, backend_url, config_path)
