"""
Cambridge Long Press Definition Test Framework
Purpose: Test agent's ability to perform long press to show inline definition
Tests: LONG PRESS action only
"""
import json
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import base64

# Browser integration imports
import sys
sys.path.append('/home/ubuntu/webarena')
from browser_env.actions import create_long_press_action, ActionTypes

# Import base configuration loader
sys.path.append('/home/ubuntu/webarena/unit_tests/IO')


class ConfigLoader:
    """Basic config loader for this test."""
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self._config = json.load(f)
    
    def get(self, key_path: str, default=None):
        keys = key_path.split('.')
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict:
        return self._config.get(section, {})


class CambridgeLongPressValidator:
    """Validator for Cambridge long press operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.long_press_executed = False
        self.press_duration = 0
        self.press_coordinates = None
        self.definition_tooltip_appeared = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_long_press_duration_and_coordinates(self, duration: int, coordinates: Dict) -> None:
        """Print long press duration and coordinates."""
        self.press_duration = duration
        self.press_coordinates = coordinates
        self.long_press_executed = True
        print(f"Long press duration and coordinates: {duration}ms at ({coordinates['x']}, {coordinates['y']})")
        self.logger.info(f"Long press executed: {duration}ms at {coordinates}")
    
    def validate_long_press_result(self) -> Dict[str, Any]:
        """Validate the long press action result."""
        validation_result = {
            'long_press_executed': self.long_press_executed,
            'duration_logged': self.press_duration > 0,
            'coordinates_logged': self.press_coordinates is not None,
            'expected_duration': 1500,
            'actual_duration': self.press_duration,
            'definition_tooltip_appeared': self.definition_tooltip_appeared,
            'long_press_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if long press was executed
        if not self.long_press_executed:
            validation_result['errors'].append("Long press action was not executed")
        
        # Check if duration was appropriate
        if self.press_duration < 1000:
            validation_result['warnings'].append(f"Long press duration may be too short: {self.press_duration}ms")
        elif self.press_duration > 2500:
            validation_result['warnings'].append(f"Long press duration may be too long: {self.press_duration}ms")
        
        # Check if coordinates were logged
        if not self.press_coordinates:
            validation_result['errors'].append("Long press coordinates were not logged")
        
        # Check if definition tooltip appeared
        if not self.definition_tooltip_appeared:
            validation_result['errors'].append("Inline definition tooltip did not appear")
        
        # Determine if long press was successful
        validation_result['long_press_successful'] = (
            self.long_press_executed and 
            self.definition_tooltip_appeared
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['long_press_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result


class CambridgeLongPressTestEnvironment:
    """Test environment for Cambridge long press with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = CambridgeLongPressValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 10)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.cambridge_elements_config = self.config_loader.get_section('cambridge_long_press_elements')
        
        self.test_state = {
            'current_step': 0,
            'long_press_executed': False,
            'word_coordinates': None,
            'definition_tooltip_before': False,
            'definition_tooltip_after': False,
            'tooltip_content': '',
            'action_history': [],
            'screenshots': [],
            'browser_session_active': False
        }
        
        # HTTP client for browser backend communication
        self.http_session = None
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_level = self.config_loader.get('logging.level', 'INFO')
        log_format = self.config_loader.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            formatter = logging.Formatter(log_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    async def get_initial_observation(self) -> Dict:
        """Get initial test setup with word 'ubiquitous' in definition text."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show Cambridge Dictionary word entry)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect word location and tooltip elements
            word_detection = await self._detect_word_elements()
            tooltip_state = await self._get_current_tooltip_state()
            
            # Store initial state
            self.test_state['word_coordinates'] = word_detection.get('word_ubiquitous', {}).get('coordinates')
            self.test_state['definition_tooltip_before'] = tooltip_state.get('tooltip_visible', False)
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_dictionary_page',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Execute long press on word "ubiquitous" to show inline definition',
                'word_detection': word_detection,
                'tooltip_state': tooltip_state,
                'initial_screenshot': screenshot_data,
                'long_press_elements': self.cambridge_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_action': 'LONG_PRESS(word_coordinates, 1500ms) once only',
                    'expected_result': 'inline definition tooltip appears'
                },
                'history': [
                    'Cambridge Dictionary page loaded',
                    'Word "ubiquitous" is visible in definition text',
                    'Word coordinates defined for long press',
                    'Inline definition tooltip should appear on long press',
                    'Ready for long press action'
                ],
                'current_dictionary': {
                    'word_coordinates': self.test_state['word_coordinates'],
                    'definition_tooltip_visible': self.test_state['definition_tooltip_before'],
                    'long_press_target_ready': True
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Cambridge long press test initialized: {self.url}")
            return setup_result
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser session: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _init_browser_session(self) -> None:
        """Initialize HTTP session and browser backend connection."""
        if not self.http_session:
            self.http_session = aiohttp.ClientSession()
        
        try:
            async with self.http_session.get(f"{self.backend_url}/health") as response:
                if response.status == 200:
                    self.test_state['browser_session_active'] = True
                    self.logger.info("Browser backend connection established")
                else:
                    raise Exception(f"Browser backend health check failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Cannot connect to browser backend at {self.backend_url}: {e}")
            raise
    
    async def _navigate_to_url(self) -> Dict:
        """Navigate browser to the test URL."""
        payload = {"url": self.url}
        
        try:
            async with self.http_session.post(f"{self.backend_url}/goto", json=payload) as response:
                result = await response.json()
                if result.get('success'):
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
            async with self.http_session.get(f"{self.backend_url}/screenshot") as response:
                if response.status == 200:
                    screenshot_bytes = await response.read()
                    screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                    return screenshot_b64
                else:
                    raise Exception(f"Screenshot request failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to get screenshot: {e}")
            raise
    
    async def _detect_word_elements(self) -> Dict:
        """Detect word location and tooltip elements."""
        detected_elements = {}
        
        # For word_ubiquitous, we use coordinates from config since finding exact word position is complex
        word_config = self.cambridge_elements_config.get('word_ubiquitous', {})
        if 'coordinates' in word_config:
            detected_elements['word_ubiquitous'] = {
                'found': True,
                'coordinates': word_config['coordinates'],
                'element_info': {
                    'word': 'ubiquitous',
                    'coordinates': word_config['coordinates']
                }
            }
        
        # Detect tooltip element
        tooltip_config = self.cambridge_elements_config.get('definition_tooltip', {})
        detected = {
            'found': False,
            'selector_used': None,
            'element_info': None
        }
        
        for selector in tooltip_config.get('selectors', []):
            try:
                eval_script = f"""
                () => {{
                    const element = document.querySelector('{selector}');
                    if (element) {{
                        const rect = element.getBoundingClientRect();
                        return {{
                            found: true,
                            selector: '{selector}',
                            visible: rect.width > 0 && rect.height > 0,
                            text_content: element.textContent || '',
                            tag: element.tagName
                        }};
                    }}
                    return {{ found: false }};
                }}
                """
                
                payload = {"script": eval_script}
                async with self.http_session.post(f"{self.backend_url}/evaluate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('found'):
                            detected = {
                                'found': True,
                                'selector_used': selector,
                                'element_info': result
                            }
                            break
                    
            except Exception as e:
                self.logger.warning(f"Failed to check selector {selector}: {e}")
                continue
        
        detected_elements['definition_tooltip'] = detected
        
        return detected_elements
    
    async def _get_current_tooltip_state(self) -> Dict:
        """Get current definition tooltip state."""
        try:
            eval_script = """
            () => {
                const result = {
                    tooltip_visible: false,
                    tooltip_content: '',
                    tooltip_count: 0
                };
                
                // Check for definition tooltips
                const tooltipElements = [
                    document.querySelector('.definition-tooltip'),
                    document.querySelector('[data-testid="inline-definition"]'),
                    document.querySelector('.word-popup'),
                    document.querySelector('.inline-popup')
                ];
                
                for (const tooltip of tooltipElements) {
                    if (tooltip) {
                        const rect = tooltip.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            result.tooltip_visible = true;
                            result.tooltip_content = tooltip.textContent || '';
                            result.tooltip_count++;
                        }
                    }
                }
                
                return result;
            }
            """
            
            payload = {"script": eval_script}
            async with self.http_session.post(f"{self.backend_url}/evaluate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to get tooltip state: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's long press action."""
        self.test_state['current_step'] += 1
        self.test_state['action_history'].append(action)
        
        evaluation = {
            'step': self.test_state['current_step'],
            'action': action,
            'valid': False,
            'feedback': '',
            'test_completed': False,
            'validation_result': None,
            'browser_result': None,
            'screenshot_after': None
        }
        
        # Check step limit (must be exactly 1 step)
        if self.test_state['current_step'] > self.max_steps:
            evaluation['feedback'] = f"Exceeded maximum steps limit ({self.max_steps}) - single action test"
            return evaluation
        
        try:
            # Execute the action in browser
            browser_result = await self._execute_action_in_browser(action)
            evaluation['browser_result'] = browser_result
            
            # Log long press duration and coordinates
            press_info = self._extract_long_press_info(action, browser_result)
            if press_info:
                self.validator.log_long_press_duration_and_coordinates(
                    press_info['duration'], 
                    press_info['coordinates']
                )
            
            # Take screenshot after action
            screenshot = await self._get_screenshot()
            evaluation['screenshot_after'] = screenshot
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': self.test_state['current_step'],
                'action': action.get('action_type'),
                'timestamp': datetime.now().isoformat(),
                'data': screenshot
            })
            
            # Get updated tooltip state
            await asyncio.sleep(0.5)  # Wait for tooltip to appear
            updated_tooltip_state = await self._get_current_tooltip_state()
            self.test_state['definition_tooltip_after'] = updated_tooltip_state.get('tooltip_visible', False)
            self.test_state['tooltip_content'] = updated_tooltip_state.get('tooltip_content', '')
            
            # Check if definition tooltip appeared
            if self.test_state['definition_tooltip_after']:
                self.validator.definition_tooltip_appeared = True
                print(f"Inline definition tooltip appears")
            
            # Evaluate the action
            action_type = action.get('action_type')
            
            if action_type == 13:  # LONG_PRESS action
                evaluation['valid'] = True
                evaluation['feedback'] = f'Long press executed - duration: {press_info["duration"] if press_info else "unknown"}ms'
                self.test_state['long_press_executed'] = True
                
                if self.test_state['definition_tooltip_after']:
                    evaluation['feedback'] += ' - inline definition tooltip successfully appeared'
                else:
                    evaluation['feedback'] += ' - inline definition tooltip did not appear'
                
                evaluation['test_completed'] = True
                evaluation['validation_result'] = self.validator.validate_long_press_result()
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected LONG_PRESS (13) only'
                
        except Exception as e:
            evaluation['feedback'] = f'Action execution failed: {str(e)}'
            self.logger.error(f"Failed to execute action: {e}")
        
        return evaluation
    
    def _extract_long_press_info(self, action: Dict, browser_result: Dict) -> Optional[Dict]:
        """Extract long press duration and coordinates information."""
        # Get coordinates
        if 'coordinates' in action:
            coordinates = action['coordinates']
        else:
            coordinates = self.cambridge_elements_config.get('word_ubiquitous', {}).get('coordinates', {'x': 0.4, 'y': 0.5})
        
        # Get duration
        duration = 1500  # Default
        if 'duration' in action:
            duration = action['duration']
        elif 'press_duration' in action:
            duration = action['press_duration']
        elif browser_result and 'long_press_duration' in browser_result:
            duration = browser_result['long_press_duration']
        
        return {
            'coordinates': coordinates,
            'duration': duration
        }
    
    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's long press action in the browser backend."""
        action_type = action.get('action_type')
        
        if action_type == 13:  # LONG_PRESS action
            # Get coordinates and duration
            coordinates = self.cambridge_elements_config.get('word_ubiquitous', {}).get('coordinates', {'x': 0.4, 'y': 0.5})
            x, y = coordinates['x'], coordinates['y']
            
            duration = 1500  # Default 1.5 seconds
            if 'duration' in action:
                duration = action['duration']
            elif 'press_duration' in action:
                duration = action['press_duration']
            
            # Execute long press by mouse down, wait, then mouse up
            payload_down = {"x": x, "y": y, "action": "mousedown"}
            payload_up = {"x": x, "y": y, "action": "mouseup"}
            
            try:
                start_time = datetime.now()
                
                # Mouse down
                async with self.http_session.post(f"{self.backend_url}/click", json=payload_down) as response:
                    await response.json()
                
                # Wait for long press duration
                await asyncio.sleep(duration / 1000.0)
                
                # Mouse up
                async with self.http_session.post(f"{self.backend_url}/click", json=payload_up) as response:
                    result = await response.json()
                
                end_time = datetime.now()
                actual_duration = (end_time - start_time).total_seconds() * 1000
                
                result.update({
                    'long_press_duration': int(actual_duration),
                    'coordinates': {'x': x, 'y': y},
                    'action_type': 'long_press'
                })
                
                self.logger.info(f"Long press executed at ({x}, {y}) for {actual_duration}ms")
                return result
            except Exception as e:
                self.logger.error(f"Long press execution failed: {e}")
                raise
        else:
            raise Exception(f"Unsupported action type for this test: {action_type}")
    
    async def cleanup(self) -> None:
        """Cleanup browser session and HTTP connections."""
        if self.http_session:
            await self.http_session.close()
            self.test_state['browser_session_active'] = False
            self.logger.info("Browser session cleanup completed")
    
    def get_test_status(self) -> Dict:
        """Get current test status."""
        return {
            'current_step': self.test_state['current_step'],
            'max_steps': self.max_steps,
            'long_press_executed': self.test_state['long_press_executed'],
            'word_coordinates': self.test_state['word_coordinates'],
            'definition_tooltip_before': self.test_state['definition_tooltip_before'],
            'definition_tooltip_after': self.test_state['definition_tooltip_after'],
            'tooltip_content': self.test_state['tooltip_content'],
            'test_completed': self.test_state['long_press_executed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'cambridge_long_press_definition',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'press_duration': self.validator.press_duration,
            'press_coordinates': self.validator.press_coordinates,
            'validation_summary': self.validator.validate_long_press_result(),
            'element_configuration': self.cambridge_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_cambridge_long_press_definition(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for Cambridge long press definition with real browser integration.
    """
    test_env = CambridgeLongPressTestEnvironment(url, backend_url, config_path)
    
    try:
        initial_obs = await test_env.get_initial_observation()
        
        if not initial_obs.get('success'):
            return {
                'success': False,
                'error': 'Failed to initialize browser environment',
                'details': initial_obs,
                'timestamp': datetime.now().isoformat()
            }
        
        agent_actions = run_agent_func(initial_obs)
        
        if not isinstance(agent_actions, list):
            await test_env.cleanup()
            return {
                'success': False,
                'error': 'Agent must return a list of actions',
                'timestamp': datetime.now().isoformat()
            }
        
        # Single action test - should only have one action
        if len(agent_actions) != 1:
            await test_env.cleanup()
            return {
                'success': False,
                'error': f'Single action test requires exactly 1 action, got {len(agent_actions)}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Evaluate the single action
        evaluation = await test_env.evaluate_agent_action(agent_actions[0])
        
        test_status = test_env.get_test_status()
        test_report = test_env.get_test_report()
        
        await test_env.cleanup()
        
        return {
            'success': evaluation['test_completed'] and evaluation['valid'],
            'initial_observation': initial_obs,
            'agent_actions': agent_actions,
            'evaluations': [evaluation],
            'test_status': test_status,
            'test_report': test_report,
            'browser_integration': True,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        await test_env.cleanup()
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# Synchronous wrapper for backward compatibility
def test_cambridge_long_press_definition_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_cambridge_long_press_definition(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_cambridge_long_press_definition_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> CambridgeLongPressTestEnvironment:
    """Factory function to create a Cambridge long press definition test environment."""
    return CambridgeLongPressTestEnvironment(url, backend_url, config_path)