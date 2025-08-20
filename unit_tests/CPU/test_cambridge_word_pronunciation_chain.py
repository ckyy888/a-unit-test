"""
Cambridge Dictionary Word Pronunciation Chain Test Framework
Purpose: Test agent's ability to compare US and UK pronunciations for ambiguous words
"""
import asyncio
import base64
import io
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


class CambridgePronunciationValidator:
    """Validator for Cambridge Dictionary pronunciation operations."""

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.audio_playback_sequence = []
        self.us_pronunciation_played = False
        self.uk_pronunciation_played = False
        self.both_pronunciations_compared = False

        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()

    def log_audio_playback_sequence(
        self, playback_type: str, audio_data: Dict
    ) -> None:
        """Log audio playback sequence."""
        playback_log = {
            "playback_id": len(self.audio_playback_sequence) + 1,
            "timestamp": datetime.now().isoformat(),
            "playback_type": playback_type,
            "audio_data": audio_data,
        }

        self.audio_playback_sequence.append(playback_log)
        self.logger.info(f"Audio playback: {playback_type}")

        if playback_type == "US_PRONUNCIATION_PLAYED":
            self.us_pronunciation_played = True
            print(
                f"Audio playback sequence: US pronunciation played for comparison"
            )
        elif playback_type == "UK_PRONUNCIATION_PLAYED":
            self.uk_pronunciation_played = True
            print(
                f"Audio playback sequence: UK pronunciation played for comparison"
            )
        elif playback_type == "AUDIO_PLAYBACK_COMPLETED":
            self.both_pronunciations_compared = True
            print(
                f"Audio playback sequence: Both pronunciation flags marked as played"
            )

    def validate_pronunciation_comparison_result(self) -> Dict[str, Any]:
        """Validate the complete pronunciation comparison sequence."""
        validation_result = {
            "us_pronunciation_played": self.us_pronunciation_played,
            "uk_pronunciation_played": self.uk_pronunciation_played,
            "both_pronunciations_compared": self.both_pronunciations_compared,
            "pronunciation_comparison_successful": False,
            "errors": [],
            "warnings": [],
        }

        # Check if US pronunciation was played
        if not self.us_pronunciation_played:
            validation_result["errors"].append(
                "US pronunciation was not played"
            )

        # Check if UK pronunciation was played
        if not self.uk_pronunciation_played:
            validation_result["errors"].append(
                "UK pronunciation was not played"
            )

        # Check if both pronunciations were compared
        if not self.both_pronunciations_compared:
            validation_result["errors"].append(
                "Both pronunciations were not compared"
            )

        # Determine if pronunciation comparison was successful
        validation_result["pronunciation_comparison_successful"] = (
            self.us_pronunciation_played
            and self.uk_pronunciation_played
            and self.both_pronunciations_compared
        )

        # Overall validation status
        validation_result["overall_valid"] = (
            validation_result["pronunciation_comparison_successful"]
            and len(validation_result["errors"]) == 0
        )

        return validation_result

    def get_playback_log(self) -> List[Dict]:
        """Return the complete log of audio playback sequence."""
        return self.audio_playback_sequence


class CambridgePronunciationTestEnvironment:
    """Test environment for Cambridge Dictionary pronunciation comparison with browser-in-browser integration."""

    def __init__(
        self,
        url: str,
        backend_url: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)

        # Always pull canonical URL from config
        self.url = self.config_loader.get("test_urls.cambridge_word_pronunciation") or url
        if not self.url:
            raise ValueError("Missing 'test_urls.cambridge_word_pronunciation' entry in config.json")
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get(
            "browser_settings.backend_url", "http://localhost:8000"
        )

        # Initialize validator with config
        self.validator = CambridgePronunciationValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)

        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get(
            "test_configuration.timeout_seconds", 30
        )
        self.screenshot_enabled = self.config_loader.get(
            "test_configuration.screenshot_enabled", True
        )
        self.cambridge_elements_config = self.config_loader.get_section(
            "cambridge_elements"
        )

        self.test_state = {
            "current_step": 0,
            "us_pronunciation_played": False,
            "uk_pronunciation_played": False,
            "both_pronunciations_compared": False,
            "us_audio_element_found": False,
            "uk_audio_element_found": False,
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
        """Get initial test setup showing search results for ambiguous word 'though'."""
        try:
            # Initialize browser session
            await self._init_browser_session()

            # Navigate to the URL (should show pronunciation variants)
            await self._navigate_to_url()

            # Take initial screenshot
            screenshot_data = await self._get_screenshot()

            # Detect pronunciation elements
            element_detection = await self._detect_pronunciation_elements()
            pronunciation_state = await self._get_current_pronunciation_state()

            # Store screenshot
            self.test_state["screenshots"].append(
                {
                    "step": "initial_pronunciation_variants",
                    "timestamp": datetime.now().isoformat(),
                    "data": screenshot_data,
                }
            )

            setup_result = {
                "success": True,
                "url": self.url,
                "task_description": "Compare US and UK pronunciations for ambiguous word",
                "element_detection": element_detection,
                "pronunciation_state": pronunciation_state,
                "initial_screenshot": screenshot_data,
                "accessibility_tree": (await self._get_accessibility_tree()).get("accessibility_tree", ""),
                "initial_accessibility_tree": (await self._get_accessibility_tree()).get("accessibility_tree", ""),
                "obs_metadata": {
                    "obs_nodes_info": (await self._get_accessibility_tree()).get("obs_nodes_info", {}),
                    "browser_config": (await self._get_accessibility_tree()).get("browser_config", {}),
                    "viewport_size": (await self._get_accessibility_tree()).get("viewport_size", {"width": 1280, "height": 800}),
                },
                "pronunciation_elements": self.cambridge_elements_config,
                "validation_rules": {
                    "max_steps": self.max_steps,
                    "required_sequence": [
                        "CLICK(US_pronunciation)",
                        "wait for audio",
                        "CLICK(UK_pronunciation)",
                    ],
                    "expected_final_state": "both pronunciation flags marked as played",
                },
                "history": [
                    "Cambridge Dictionary search page loaded",
                    'Searched for "though" (ambiguous word)',
                    "Multiple pronunciation variants shown",
                    "US and UK pronunciation icons visible",
                    "Need to play both audio samples for comparison",
                ],
                "current_word": {
                    "word": "though",
                    "pronunciation_variants": ["US", "UK"],
                    "us_played": False,
                    "uk_played": False,
                },
                "current_step": 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Update test state based on detection
            if element_detection.get("us_pronunciation", {}).get("found"):
                self.test_state["us_audio_element_found"] = True
            if element_detection.get("uk_pronunciation", {}).get("found"):
                self.test_state["uk_audio_element_found"] = True

            self.logger.info(
                f"Cambridge pronunciation comparison test initialized: {self.url}"
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
        """Get current screenshot using WebSocket when available (B-i-B alignment)."""
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
                draw = ImageDraw.Draw(img)
                draw.text((100, 100), "Cambridge Pronunciation (mock)", fill="black")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                return "screenshot_generation_failed"

    async def _detect_pronunciation_elements(self) -> Dict:
        """Detect pronunciation audio elements on the current page."""
        detected_elements = {}

        for element_type, config in self.cambridge_elements_config.items():
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
                                aria_label: element.getAttribute('aria-label') || '',
                                data_region: element.getAttribute('data-region') || '',
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

    async def _get_current_pronunciation_state(self) -> Dict:
        """Get current pronunciation state and audio playback status."""
        try:
            eval_script = """
            () => {
                const result = {
                    us_pronunciation_available: false,
                    uk_pronunciation_available: false,
                    us_played: false,
                    uk_played: false,
                    word_displayed: '',
                    phonetic_text: ''
                };

                // Check for US pronunciation elements
                const usElements = [
                    document.querySelector('.us-pronunciation'),
                    document.querySelector('[data-region="us"] .audio-play'),
                    document.querySelector('button[aria-label*="US pronunciation"]')
                ];

                for (const element of usElements) {
                    if (element) {
                        result.us_pronunciation_available = true;
                        // Check if audio has been played (common patterns)
                        if (element.classList.contains('played') || element.getAttribute('aria-pressed') === 'true') {
                            result.us_played = true;
                        }
                        break;
                    }
                }

                // Check for UK pronunciation elements
                const ukElements = [
                    document.querySelector('.uk-pronunciation'),
                    document.querySelector('[data-region="uk"] .audio-play'),
                    document.querySelector('button[aria-label*="UK pronunciation"]')
                ];

                for (const element of ukElements) {
                    if (element) {
                        result.uk_pronunciation_available = true;
                        // Check if audio has been played
                        if (element.classList.contains('played') || element.getAttribute('aria-pressed') === 'true') {
                            result.uk_played = true;
                        }
                        break;
                    }
                }

                // Get word being pronounced
                const wordElement = document.querySelector('.word-title, .headword, h1');
                if (wordElement) {
                    result.word_displayed = wordElement.textContent.trim();
                }

                // Get phonetic text if available
                const phoneticElement = document.querySelector('.phonetic, .ipa, [class*="pronunciation"]');
                if (phoneticElement) {
                    result.phonetic_text = phoneticElement.textContent.trim();
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
            self.logger.error(f"Failed to get pronunciation state: {e}")
            return {}

    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's pronunciation action."""
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

            # Get updated pronunciation state
            pronunciation_state = await self._get_current_pronunciation_state()

            # Evaluate the action
            action_type = action.get("action_type")
            element_name = action.get("element_name", "").lower()

            if action_type == 6:  # CLICK action
                if "us" in element_name and "pronunciation" in element_name:
                    # US pronunciation clicked
                    evaluation["valid"] = True
                    evaluation[
                        "feedback"
                    ] = "US pronunciation button clicked - audio playing"
                    self.test_state["us_pronunciation_played"] = True

                    # Log the pronunciation action
                    self.validator.log_audio_playback_sequence(
                        "US_PRONUNCIATION_PLAYED",
                        {
                            "region": "US",
                            "audio_started": True,
                            "word": pronunciation_state.get(
                                "word_displayed", "though"
                            ),
                        },
                    )

                    if not self.test_state["uk_pronunciation_played"]:
                        evaluation[
                            "expected_next"
                        ] = "Wait for audio to complete, then click UK pronunciation"
                    else:
                        # Both pronunciations played
                        self._complete_pronunciation_comparison()
                        evaluation["test_completed"] = True
                        evaluation[
                            "validation_result"
                        ] = (
                            self.validator.validate_pronunciation_comparison_result()
                        )

                elif "uk" in element_name and "pronunciation" in element_name:
                    # UK pronunciation clicked
                    evaluation["valid"] = True
                    evaluation[
                        "feedback"
                    ] = "UK pronunciation button clicked - audio playing"
                    self.test_state["uk_pronunciation_played"] = True

                    # Log the pronunciation action
                    self.validator.log_audio_playback_sequence(
                        "UK_PRONUNCIATION_PLAYED",
                        {
                            "region": "UK",
                            "audio_started": True,
                            "word": pronunciation_state.get(
                                "word_displayed", "though"
                            ),
                        },
                    )

                    if not self.test_state["us_pronunciation_played"]:
                        evaluation[
                            "expected_next"
                        ] = "Click US pronunciation to complete comparison"
                    else:
                        # Both pronunciations played
                        self._complete_pronunciation_comparison()
                        evaluation["test_completed"] = True
                        evaluation[
                            "validation_result"
                        ] = (
                            self.validator.validate_pronunciation_comparison_result()
                        )

                else:
                    evaluation[
                        "feedback"
                    ] = f"Clicked wrong element - expected pronunciation button, got: {element_name}"
            else:
                evaluation[
                    "feedback"
                ] = f"Unexpected action type {action_type} - expected CLICK (6)"

        except Exception as e:
            evaluation["feedback"] = f"Action execution failed: {str(e)}"
            self.logger.error(f"Failed to execute action: {e}")

        return evaluation

    def _complete_pronunciation_comparison(self) -> None:
        """Mark pronunciation comparison as complete."""
        self.test_state["both_pronunciations_compared"] = True

        # Log the completion
        self.validator.log_audio_playback_sequence(
            "AUDIO_PLAYBACK_COMPLETED",
            {
                "us_played": self.test_state["us_pronunciation_played"],
                "uk_played": self.test_state["uk_pronunciation_played"],
                "comparison_complete": True,
            },
        )

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
                    if "us" in element_name and "pronunciation" in element_name:
                        coords = self.cambridge_elements_config.get("us_pronunciation", {}).get("coordinates", {})
                        x, y = coords.get("x", 0.3), coords.get("y", 0.6)
                    elif "uk" in element_name and "pronunciation" in element_name:
                        coords = self.cambridge_elements_config.get("uk_pronunciation", {}).get("coordinates", {})
                        x, y = coords.get("x", 0.7), coords.get("y", 0.6)
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

                    # Add small delay for audio to start
                    await asyncio.sleep(1)

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
            "us_pronunciation_played": self.test_state[
                "us_pronunciation_played"
            ],
            "uk_pronunciation_played": self.test_state[
                "uk_pronunciation_played"
            ],
            "both_pronunciations_compared": self.test_state[
                "both_pronunciations_compared"
            ],
            "test_completed": self.test_state["both_pronunciations_compared"],
            "action_history": self.test_state["action_history"],
            "steps_remaining": self.max_steps
            - self.test_state["current_step"],
        }

    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            "test_type": "cambridge_word_pronunciation_chain",
            "test_url": self.url,
            "max_steps_limit": self.max_steps,
            "audio_playback_sequence": self.validator.get_playback_log(),
            "validation_summary": self.validator.validate_pronunciation_comparison_result(),
            "element_configuration": self.cambridge_elements_config,
            "timestamp": datetime.now().isoformat(),
        }


# Test runner function for external agent systems
async def test_cambridge_word_pronunciation_chain(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Test function for Cambridge Dictionary pronunciation comparison with real browser integration.
    """
    test_env = CambridgePronunciationTestEnvironment(
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
def test_cambridge_word_pronunciation_chain_sync(
    url: str,
    run_agent_func: Callable,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(
        test_cambridge_word_pronunciation_chain(
            url, run_agent_func, backend_url, config_path
        )
    )


# Convenience function for quick integration
def create_cambridge_word_pronunciation_chain_test(
    url: str,
    backend_url: Optional[str] = None,
    config_path: Optional[str] = None,
) -> CambridgePronunciationTestEnvironment:
    """Factory function to create a Cambridge pronunciation comparison test environment."""
    return CambridgePronunciationTestEnvironment(url, backend_url, config_path)
