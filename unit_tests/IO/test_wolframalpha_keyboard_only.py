"""
WolframAlpha Keyboard Only Test Framework
Purpose: Test agent's ability to perform precise keyboard input without key presses
Tests: KEYBOARD EXECUTION (TYPE) action only
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
from browser_env.actions import create_keyboard_type_action, ActionTypes

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


class WolframAlphaKeyboardValidator:
    """Validator for WolframAlpha keyboard operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.text_entered = False
        self.exact_text = None
        self.query_box_contains_text = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_exact_text_entered(self, text: str) -> None:
        """Print exact text entered."""
        self.exact_text = text
        self.text_entered = True
        print(f"Exact text entered: '{text}'")
        self.logger.info(f"Keyboard input executed: '{text}'")
    
    def validate_keyboard_result(self) -> Dict[str, Any]:
        """Validate the keyboard input result."""
        validation_result = {
            'text_entered': self.text_entered,
            'exact_text_logged': self.exact_text is not None,
            'expected_text': '2+2',
            'actual_text': self.exact_text,
            'query_box_contains_text': self.query_box_contains_text,
            'text_matches_expected': False,
            'keyboard_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if text was entered
        if not self.text_entered:
            validation_result['errors'].append("Keyboard input was not executed")
        
        # Check if exact text was logged
        if not self.exact_text:
            validation_result['errors'].append("Exact text entered was not logged")
        
        # Check if text matches expected
        if self.exact_text == '2+2':
            validation_result['text_matches_expected'] = True
        else:
            validation_result['errors'].append(f"Text does not match expected '2+2', got '{self.exact_text}'")
        
        # Check if query box contains the text
        if not self.query_box_contains_text:
            validation_result['errors'].append("Query box does not contain '2+2' text")
        
        # Determine if keyboard input was successful
        validation_result['keyboard_successful'] = (
            self.text_entered and 
            self.exact_text == '2+2' and
            self.query_box_contains_text
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['keyboard_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result


class WolframAlphaKeyboardTestEnvironment:
    """Test environment for WolframAlpha keyboard with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = WolframAlphaKeyboardValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 10)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.wolfram_elements_config = self.config_loader.get_section('wolframalpha_keyboard_elements')
        
        self.test_state = {
            'current_step': 0,
            'keyboard_executed': False,
            'text_entered': '',
            'query_box_focused': False,
            'query_box_empty_before': True,
            'query_box_contains_text_after': False,
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
        """Get initial test setup with cursor active in query box, empty field."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show WolframAlpha homepage)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect query box and focus state
            query_detection = await self._detect_query_box()
            input_state = await self._get_current_input_state()
            
            # Store initial input state
            self.test_state['query_box_focused'] = input_state.get('focused', False)
            self.test_state['query_box_empty_before'] = input_state.get('empty', True)
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_empty_query_box',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Execute keyboard input "2+2" without any key presses',
                'query_detection': query_detection,
                'input_state': input_state,
                'initial_screenshot': screenshot_data,
                'keyboard_elements': self.wolfram_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_action': 'TYPE("2+2") without any key presses',
                    'expected_result': 'query box contains "2+2" text'
                },
                'history': [
                    'WolframAlpha homepage loaded',
                    'Query box is active and focused',
                    'Field is empty and ready for input',
                    'Cursor is positioned in input field',
                    'Ready for keyboard input action'
                ],
                'current_input': {
                    'focused': self.test_state['query_box_focused'],
                    'empty': self.test_state['query_box_empty_before'],
                    'ready_for_input': True
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"WolframAlpha keyboard test initialized: {self.url}")
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
    
    async def _detect_query_box(self) -> Dict:
        """Detect query input box and its state."""
        query_config = self.wolfram_elements_config.get('query_box', {})
        detected = {
            'found': False,
            'selector_used': None,
            'coordinates': None,
            'element_info': None
        }
        
        for selector in query_config.get('selectors', []):
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
                            focused: document.activeElement === element,
                            enabled: !element.disabled,
                            value: element.value || '',
                            placeholder: element.placeholder || '',
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
                                'coordinates': result.get('coordinates'),
                                'element_info': result
                            }
                            break
                    
            except Exception as e:
                self.logger.warning(f"Failed to check selector {selector}: {e}")
                continue
        
        return detected
    
    async def _get_current_input_state(self) -> Dict:
        """Get current input field state."""
        try:
            eval_script = """
            () => {
                const result = {
                    focused: false,
                    empty: true,
                    current_value: '',
                    cursor_position: 0
                };
                
                const queryBox = document.querySelector('#input') ||
                                document.querySelector('[data-testid="query-input"]') ||
                                document.querySelector('.query-field') ||
                                document.querySelector('input[type="text"][placeholder*="query"]');
                
                if (queryBox) {
                    result.focused = document.activeElement === queryBox;
                    result.current_value = queryBox.value || '';
                    result.empty = result.current_value.trim().length === 0;
                    result.cursor_position = queryBox.selectionStart || 0;
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
            self.logger.error(f"Failed to get input state: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's keyboard action."""
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
            
            # Log exact text entered
            text_entered = action.get('text', '')
            if text_entered:
                self.validator.log_exact_text_entered(text_entered)
                self.test_state['text_entered'] = text_entered
            
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
            
            # Get updated input state
            await asyncio.sleep(0.5)  # Wait for input to be processed
            updated_input_state = await self._get_current_input_state()
            
            current_value = updated_input_state.get('current_value', '')
            self.test_state['query_box_contains_text_after'] = '2+2' in current_value
            
            # Evaluate the action
            action_type = action.get('action_type')
            
            if action_type == 3:  # TYPE action
                evaluation['valid'] = True
                evaluation['feedback'] = f'Keyboard input executed: "{text_entered}"'
                self.test_state['keyboard_executed'] = True
                
                # Check if query box contains the expected text
                if self.test_state['query_box_contains_text_after']:
                    self.validator.query_box_contains_text = True
                    print(f'Query box contains "2+2" text')
                    evaluation['feedback'] += ' - query box successfully updated'
                else:
                    evaluation['feedback'] += f' - query box does not contain expected text (got: "{current_value}")'
                
                evaluation['test_completed'] = True
                evaluation['validation_result'] = self.validator.validate_keyboard_result()
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected TYPE (3) only'
                
        except Exception as e:
            evaluation['feedback'] = f'Action execution failed: {str(e)}'
            self.logger.error(f"Failed to execute action: {e}")
        
        return evaluation
    
    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's action in the browser backend."""
        action_type = action.get('action_type')
        
        if action_type == 3:  # TYPE action
            text = action.get('text', '')
            
            # Execute text input via browser backend
            payload = {"key": text}
            
            try:
                async with self.http_session.post(f"{self.backend_url}/keyboard", json=payload) as response:
                    result = await response.json()
                    self.logger.info(f"Keyboard input executed: '{text}'")
                    return result
            except Exception as e:
                self.logger.error(f"Keyboard input failed: {e}")
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
            'keyboard_executed': self.test_state['keyboard_executed'],
            'text_entered': self.test_state['text_entered'],
            'query_box_focused': self.test_state['query_box_focused'],
            'query_box_empty_before': self.test_state['query_box_empty_before'],
            'query_box_contains_text_after': self.test_state['query_box_contains_text_after'],
            'test_completed': self.test_state['keyboard_executed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'wolframalpha_keyboard_only',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'text_entered': self.validator.exact_text,
            'validation_summary': self.validator.validate_keyboard_result(),
            'element_configuration': self.wolfram_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_wolframalpha_keyboard_only(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for WolframAlpha keyboard only with real browser integration.
    """
    test_env = WolframAlphaKeyboardTestEnvironment(url, backend_url, config_path)
    
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
def test_wolframalpha_keyboard_only_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_wolframalpha_keyboard_only(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_wolframalpha_keyboard_only_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> WolframAlphaKeyboardTestEnvironment:
    """Factory function to create a WolframAlpha keyboard only test environment."""
    return WolframAlphaKeyboardTestEnvironment(url, backend_url, config_path)