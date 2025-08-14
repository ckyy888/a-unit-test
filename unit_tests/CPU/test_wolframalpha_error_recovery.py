"""
WolframAlpha Error Recovery Test Framework
Purpose: Test agent's ability to fix malformed equation and get valid mathematical result
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
from browser_env.actions import create_click_action, create_keyboard_type_action, ActionTypes

# Import base configuration loader
sys.path.append('/home/ubuntu/webarena/unit_tests/CPU')


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


class WolframAlphaValidator:
    """Validator for WolframAlpha error recovery operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.query_corrections = []
        self.query_cleared = False
        self.corrected_query_entered = False
        self.valid_result_obtained = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_query_correction(self, correction_type: str, query_data: Dict) -> None:
        """Log query corrections."""
        correction_log = {
            'correction_id': len(self.query_corrections) + 1,
            'timestamp': datetime.now().isoformat(),
            'correction_type': correction_type,
            'query_data': query_data
        }
        
        self.query_corrections.append(correction_log)
        self.logger.info(f"Query correction: {correction_type}")
        
        if correction_type == 'QUERY_BOX_CLICKED':
            print(f"Query corrections: Query box focused for editing")
        elif correction_type == 'FIELD_CLEARED':
            self.query_cleared = True
            print(f"Query corrections: Field cleared of malformed equation")
        elif correction_type == 'CORRECTED_QUERY_ENTERED':
            self.corrected_query_entered = True
            print(f"Query corrections: New query entered - {query_data.get('corrected_query', '2+2*3')}")
        elif correction_type == 'VALID_RESULT_OBTAINED':
            self.valid_result_obtained = True
            print(f"Query corrections: Valid mathematical result obtained - {query_data.get('result', 'calculation complete')}")
    
    def validate_error_recovery_result(self) -> Dict[str, Any]:
        """Validate the complete error recovery sequence."""
        validation_result = {
            'query_cleared': self.query_cleared,
            'corrected_query_entered': self.corrected_query_entered,
            'valid_result_obtained': self.valid_result_obtained,
            'recovery_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if query was cleared
        if not self.query_cleared:
            validation_result['errors'].append("Malformed query was not cleared from field")
        
        # Check if corrected query was entered
        if not self.corrected_query_entered:
            validation_result['errors'].append("Corrected query (2+2*3) was not entered")
        
        # Check if valid result was obtained
        if not self.valid_result_obtained:
            validation_result['errors'].append("Valid mathematical result was not obtained")
        
        # Determine if recovery was successful
        validation_result['recovery_successful'] = (
            self.query_cleared and 
            self.corrected_query_entered and 
            self.valid_result_obtained
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['recovery_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    def get_correction_log(self) -> List[Dict]:
        """Return the complete log of query corrections."""
        return self.query_corrections


class WolframAlphaTestEnvironment:
    """Test environment for WolframAlpha error recovery with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = WolframAlphaValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 30)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.wolfram_elements_config = self.config_loader.get_section('wolframalpha_elements')
        
        self.test_state = {
            'current_step': 0,
            'malformed_query': '2+2*',  # Starting malformed query
            'corrected_query': '2+2*3',  # Target corrected query
            'query_box_focused': False,
            'field_cleared': False,
            'corrected_query_entered': False,
            'valid_result_displayed': False,
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
        """Get initial test setup with error state visible."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show "No results found" error)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect query elements and error state
            element_detection = await self._detect_query_elements()
            error_state = await self._check_error_state()
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_error_state',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Fix malformed equation by clearing field and entering corrected query',
                'element_detection': element_detection,
                'error_state': error_state,
                'initial_screenshot': screenshot_data,
                'query_elements': self.wolfram_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_sequence': ['CLICK(query_box)', 'CLEAR_FIELD', 'TYPE("2+2*3")'],
                    'expected_final_state': 'valid mathematical result appears'
                },
                'history': [
                    'WolframAlpha results page loaded',
                    'Searched for "2+2*" (malformed equation)',
                    'Got "No results found" error',
                    'Query box contains malformed equation',
                    'Need to fix syntax error'
                ],
                'current_query': {
                    'text': self.test_state['malformed_query'],
                    'malformed': True,
                    'error_displayed': True
                },
                'target_query': {
                    'text': self.test_state['corrected_query'],
                    'expected_result': '8'
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update test state based on detection
            if element_detection.get('query_box', {}).get('found'):
                self.test_state['query_box_available'] = True
            if error_state.get('error_found'):
                self.test_state['error_displayed'] = True
            
            self.logger.info(f"WolframAlpha error recovery test initialized: {self.url}")
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
    
    async def _detect_query_elements(self) -> Dict:
        """Detect query input elements on the current page."""
        detected_elements = {}
        
        for element_type, config in self.wolfram_elements_config.items():
            detected = {
                'found': False,
                'selector_used': None,
                'coordinates': None,
                'element_info': None
            }
            
            # Try each selector to find the element
            for selector in config.get('selectors', []):
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
            
            detected_elements[element_type] = detected
        
        return detected_elements
    
    async def _check_error_state(self) -> Dict:
        """Check if error message is displayed."""
        try:
            eval_script = """
            () => {
                const result = {
                    error_found: false,
                    error_message: '',
                    no_results_displayed: false
                };
                
                // Look for "No results found" or similar error messages
                const errorSelectors = [
                    'text=No results found',
                    '.error-message',
                    '.no-results',
                    '[data-testid="error"]'
                ];
                
                for (const selector of errorSelectors) {
                    const elements = document.querySelectorAll('*');
                    for (const element of elements) {
                        if (element.textContent && element.textContent.toLowerCase().includes('no results')) {
                            result.error_found = true;
                            result.no_results_displayed = true;
                            result.error_message = element.textContent;
                            break;
                        }
                    }
                    if (result.error_found) break;
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
                    return {'error_found': False}
        except Exception as e:
            self.logger.error(f"Failed to check error state: {e}")
            return {'error_found': False}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's error recovery action."""
        self.test_state['current_step'] += 1
        self.test_state['action_history'].append(action)
        
        evaluation = {
            'step': self.test_state['current_step'],
            'action': action,
            'valid': False,
            'feedback': '',
            'expected_next': None,
            'test_completed': False,
            'validation_result': None,
            'browser_result': None,
            'screenshot_after': None
        }
        
        # Check step limit
        if self.test_state['current_step'] > self.max_steps:
            evaluation['feedback'] = f"Exceeded maximum steps limit ({self.max_steps})"
            return evaluation
        
        try:
            # Execute the action in browser
            browser_result = await self._execute_action_in_browser(action)
            evaluation['browser_result'] = browser_result
            
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
            
            # Get current query state
            query_state = await self._get_current_query_state()
            
            # Evaluate the action
            action_type = action.get('action_type')
            element_name = action.get('element_name', '').lower()
            text_content = action.get('text', '')
            
            if action_type == 6:  # CLICK action
                if 'query' in element_name or 'input' in element_name:
                    # Query box clicked for editing
                    evaluation['valid'] = True
                    evaluation['feedback'] = 'Query box focused for editing'
                    self.test_state['query_box_focused'] = True
                    
                    # Log the correction step
                    self.validator.log_query_correction('QUERY_BOX_CLICKED', {
                        'focused': True,
                        'current_query': query_state.get('current_query', '')
                    })
                    
                    evaluation['expected_next'] = 'Clear the field and enter corrected query'
                else:
                    evaluation['feedback'] = f'Clicked wrong element - expected query box, got: {element_name}'
            
            elif action_type == 3:  # TYPE action
                if 'clear' in text_content.lower() or len(text_content.strip()) == 0:
                    # Field clearing action
                    evaluation['valid'] = True
                    evaluation['feedback'] = 'Field cleared of malformed equation'
                    self.test_state['field_cleared'] = True
                    
                    # Log the correction step
                    self.validator.log_query_correction('FIELD_CLEARED', {
                        'previous_query': self.test_state['malformed_query'],
                        'cleared': True
                    })
                    
                    evaluation['expected_next'] = 'Enter corrected query (2+2*3)'
                
                elif '2+2*3' in text_content:
                    # Corrected query entered
                    evaluation['valid'] = True
                    evaluation['feedback'] = 'Corrected query (2+2*3) entered successfully'
                    self.test_state['corrected_query_entered'] = True
                    
                    # Log the correction step
                    self.validator.log_query_correction('CORRECTED_QUERY_ENTERED', {
                        'corrected_query': text_content,
                        'expected_result': '8'
                    })
                    
                    # Check if valid result appears
                    result_state = await self._check_valid_result()
                    if result_state.get('valid_result_found'):
                        self.test_state['valid_result_displayed'] = True
                        
                        # Log successful recovery
                        self.validator.log_query_correction('VALID_RESULT_OBTAINED', {
                            'result': result_state.get('result_value', '8'),
                            'recovery_successful': True
                        })
                        
                        evaluation['test_completed'] = True
                        evaluation['validation_result'] = self.validator.validate_error_recovery_result()
                    else:
                        evaluation['expected_next'] = 'Wait for valid mathematical result to appear'
                
                else:
                    evaluation['feedback'] = f'Unexpected text entered - expected "2+2*3", got: {text_content}'
            
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected CLICK (6) or TYPE (3)'
                
        except Exception as e:
            evaluation['feedback'] = f'Action execution failed: {str(e)}'
            self.logger.error(f"Failed to execute action: {e}")
        
        return evaluation
    
    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's action in the browser backend."""
        action_type = action.get('action_type')
        
        if action_type == 6:  # CLICK action
            # Determine click coordinates
            if 'coordinates' in action:
                x, y = action['coordinates']['x'], action['coordinates']['y']
            elif 'element_name' in action:
                element_name = action['element_name'].lower()
                if 'query' in element_name or 'input' in element_name:
                    coords = self.wolfram_elements_config.get('query_box', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.5), coords.get('y', 0.2)
                else:
                    x, y = 0.5, 0.5  # Default center click
            else:
                x, y = 0.5, 0.5  # Default center click
            
            # Execute click via browser backend
            payload = {"x": x, "y": y}
            
            try:
                async with self.http_session.post(f"{self.backend_url}/click", json=payload) as response:
                    result = await response.json()
                    self.logger.info(f"Click executed at ({x}, {y}): {result}")
                    return result
            except Exception as e:
                self.logger.error(f"Click execution failed: {e}")
                raise
        
        elif action_type == 3:  # TYPE action
            text = action.get('text', '')
            
            # Handle special clear field action
            if 'clear' in text.lower():
                # Clear field by selecting all and typing empty
                clear_script = """
                () => {
                    const input = document.querySelector('#input') || 
                                document.querySelector('[data-testid="query-input"]') ||
                                document.querySelector('.query-field');
                    if (input) {
                        input.focus();
                        input.select();
                        input.value = '';
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        return { success: true, cleared: true };
                    }
                    return { success: false };
                }
                """
                
                payload = {"script": clear_script}
                try:
                    async with self.http_session.post(f"{self.backend_url}/evaluate", json=payload) as response:
                        result = await response.json()
                        self.logger.info(f"Field cleared: {result}")
                        return result
                except Exception as e:
                    self.logger.error(f"Field clear failed: {e}")
                    raise
            else:
                # Regular text input
                payload = {"key": text}
                
                try:
                    async with self.http_session.post(f"{self.backend_url}/keyboard", json=payload) as response:
                        result = await response.json()
                        self.logger.info(f"Text typed: {text}")
                        return result
                except Exception as e:
                    self.logger.error(f"Text input failed: {e}")
                    raise
        else:
            raise Exception(f"Unsupported action type for this test: {action_type}")
    
    async def _get_current_query_state(self) -> Dict:
        """Get the current query input state."""
        try:
            eval_script = """
            () => {
                const result = {
                    current_query: '',
                    focused: false,
                    empty: true
                };
                
                const input = document.querySelector('#input') || 
                            document.querySelector('[data-testid="query-input"]') ||
                            document.querySelector('.query-field');
                
                if (input) {
                    result.current_query = input.value || '';
                    result.focused = document.activeElement === input;
                    result.empty = result.current_query.trim().length === 0;
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
            self.logger.error(f"Failed to get query state: {e}")
            return {}
    
    async def _check_valid_result(self) -> Dict:
        """Check if a valid mathematical result is displayed."""
        try:
            eval_script = """
            () => {
                const result = {
                    valid_result_found: false,
                    result_value: '',
                    calculation_complete: false
                };
                
                // Look for result pods or calculation results
                const resultSelectors = [
                    '#results',
                    '.result-pod',
                    '[data-testid="result-container"]',
                    '.output'
                ];
                
                for (const selector of resultSelectors) {
                    const element = document.querySelector(selector);
                    if (element && element.textContent) {
                        const text = element.textContent;
                        // Look for number results (like "8" for 2+2*3)
                        if (/^\s*8\s*$/.test(text) || text.includes('= 8') || text.includes('8')) {
                            result.valid_result_found = true;
                            result.result_value = '8';
                            result.calculation_complete = true;
                            break;
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
                    return {'valid_result_found': False}
        except Exception as e:
            self.logger.error(f"Failed to check valid result: {e}")
            return {'valid_result_found': False}
    
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
            'query_box_focused': self.test_state['query_box_focused'],
            'field_cleared': self.test_state['field_cleared'],
            'corrected_query_entered': self.test_state['corrected_query_entered'],
            'valid_result_displayed': self.test_state['valid_result_displayed'],
            'test_completed': self.test_state['valid_result_displayed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'wolframalpha_error_recovery',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'query_corrections': self.validator.get_correction_log(),
            'validation_summary': self.validator.validate_error_recovery_result(),
            'element_configuration': self.wolfram_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_wolframalpha_error_recovery(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for WolframAlpha error recovery with real browser integration.
    """
    test_env = WolframAlphaTestEnvironment(url, backend_url, config_path)
    
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
        
        evaluations = []
        test_completed = False
        
        for action in agent_actions:
            evaluation = await test_env.evaluate_agent_action(action)
            evaluations.append(evaluation)
            
            if evaluation['test_completed']:
                test_completed = True
                break
        
        test_status = test_env.get_test_status()
        test_report = test_env.get_test_report()
        
        await test_env.cleanup()
        
        return {
            'success': test_completed and test_status['test_completed'],
            'initial_observation': initial_obs,
            'agent_actions': agent_actions,
            'evaluations': evaluations,
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
def test_wolframalpha_error_recovery_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_wolframalpha_error_recovery(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_wolframalpha_error_recovery_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> WolframAlphaTestEnvironment:
    """Factory function to create a WolframAlpha error recovery test environment."""
    return WolframAlphaTestEnvironment(url, backend_url, config_path)