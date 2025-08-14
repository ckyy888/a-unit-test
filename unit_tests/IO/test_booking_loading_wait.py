"""
Booking.com Loading Wait Test Framework
Purpose: Test agent's ability to wait for search results loading to complete
Tests: WAIT action only
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
from browser_env.actions import create_wait_action, ActionTypes

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


class BookingWaitValidator:
    """Validator for Booking.com wait operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.wait_executed = False
        self.wait_duration = 0
        self.spinner_disappeared = False
        self.hotel_results_appeared = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_wait_duration_executed(self, duration: int) -> None:
        """Print wait duration executed."""
        self.wait_duration = duration
        self.wait_executed = True
        print(f"Wait duration executed: {duration}ms")
        self.logger.info(f"Wait executed for duration: {duration}ms")
    
    def validate_wait_result(self) -> Dict[str, Any]:
        """Validate the wait action result."""
        validation_result = {
            'wait_executed': self.wait_executed,
            'wait_duration_logged': self.wait_duration > 0,
            'expected_duration': 3000,
            'actual_duration': self.wait_duration,
            'spinner_disappeared': self.spinner_disappeared,
            'hotel_results_appeared': self.hotel_results_appeared,
            'loading_completed': False,
            'wait_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if wait was executed
        if not self.wait_executed:
            validation_result['errors'].append("Wait action was not executed")
        
        # Check if wait duration is appropriate
        if self.wait_duration < 2000:
            validation_result['warnings'].append(f"Wait duration may be insufficient: {self.wait_duration}ms")
        elif self.wait_duration > 5000:
            validation_result['warnings'].append(f"Wait duration may be excessive: {self.wait_duration}ms")
        
        # Check if spinner disappeared
        if not self.spinner_disappeared:
            validation_result['errors'].append("Loading spinner did not disappear")
        
        # Check if hotel results appeared
        if not self.hotel_results_appeared:
            validation_result['errors'].append("Hotel results did not appear")
        
        # Determine loading completion
        validation_result['loading_completed'] = (
            self.spinner_disappeared and 
            self.hotel_results_appeared
        )
        
        # Determine if wait was successful
        validation_result['wait_successful'] = (
            self.wait_executed and 
            validation_result['loading_completed']
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['wait_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result


class BookingWaitTestEnvironment:
    """Test environment for Booking.com wait with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = BookingWaitValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 10)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.booking_elements_config = self.config_loader.get_section('booking_wait_elements')
        
        self.test_state = {
            'current_step': 0,
            'wait_executed': False,
            'wait_duration': 0,
            'spinner_visible_before': True,
            'spinner_visible_after': False,
            'results_visible_before': False,
            'results_visible_after': False,
            'loading_completed': False,
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
        """Get initial test setup showing 'Searching...' spinner after submitting search."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show Booking.com with search loading)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect spinner and results area
            loading_detection = await self._detect_loading_elements()
            loading_state = await self._get_current_loading_state()
            
            # Store initial loading state
            self.test_state['spinner_visible_before'] = loading_state.get('spinner_visible', True)
            self.test_state['results_visible_before'] = loading_state.get('results_visible', False)
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_search_loading',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Wait for search results loading to complete',
                'loading_detection': loading_detection,
                'loading_state': loading_state,
                'initial_screenshot': screenshot_data,
                'wait_elements': self.booking_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_action': 'WAIT(3000ms) for results to load',
                    'expected_result': 'spinner disappears and hotel results appear'
                },
                'history': [
                    'Booking.com search page loaded',
                    'Hotel search submitted',
                    '"Searching..." spinner is active',
                    'Search is in progress',
                    'Need to wait for results to load'
                ],
                'current_loading': {
                    'spinner_visible': self.test_state['spinner_visible_before'],
                    'results_visible': self.test_state['results_visible_before'],
                    'loading_in_progress': True
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Booking.com wait test initialized: {self.url}")
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
    
    async def _detect_loading_elements(self) -> Dict:
        """Detect loading spinner and results area elements."""
        detected_elements = {}
        
        for element_type, config in self.booking_elements_config.items():
            detected = {
                'found': False,
                'selector_used': None,
                'visible': False,
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
                                visible: rect.width > 0 && rect.height > 0,
                                display_style: window.getComputedStyle(element).display,
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
                                    'visible': result.get('visible', False),
                                    'element_info': result
                                }
                                break
                        
                except Exception as e:
                    self.logger.warning(f"Failed to check selector {selector}: {e}")
                    continue
            
            detected_elements[element_type] = detected
        
        return detected_elements
    
    async def _get_current_loading_state(self) -> Dict:
        """Get current loading state - spinner and results visibility."""
        try:
            eval_script = """
            () => {
                const result = {
                    spinner_visible: false,
                    results_visible: false,
                    loading_complete: false,
                    results_count: 0
                };
                
                // Check loading spinner
                const spinnerElements = [
                    document.querySelector('.loading-spinner'),
                    document.querySelector('[data-testid="loading"]'),
                    document.querySelector('.spinner'),
                    document.querySelector('.searching-indicator')
                ];
                
                for (const element of spinnerElements) {
                    if (element) {
                        const rect = element.getBoundingClientRect();
                        const isVisible = rect.width > 0 && rect.height > 0 && 
                                        window.getComputedStyle(element).display !== 'none';
                        if (isVisible) {
                            result.spinner_visible = true;
                            break;
                        }
                    }
                }
                
                // Check results area
                const resultsElements = [
                    document.querySelector('.search-results'),
                    document.querySelector('[data-testid="hotel-results"]'),
                    document.querySelector('.results-container')
                ];
                
                for (const element of resultsElements) {
                    if (element) {
                        const rect = element.getBoundingClientRect();
                        const isVisible = rect.width > 0 && rect.height > 0;
                        if (isVisible) {
                            result.results_visible = true;
                            // Count hotel results
                            const hotelCards = element.querySelectorAll('.hotel-card, .property-card, [data-testid="property"]');
                            result.results_count = hotelCards.length;
                            break;
                        }
                    }
                }
                
                // Determine if loading is complete
                result.loading_complete = !result.spinner_visible && result.results_visible;
                
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
            self.logger.error(f"Failed to get loading state: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's wait action."""
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
            # Execute the action (wait)
            wait_duration = self._extract_wait_duration(action)
            browser_result = await self._execute_action_in_browser(action)
            evaluation['browser_result'] = browser_result
            
            # Log wait duration
            self.validator.log_wait_duration_executed(wait_duration)
            self.test_state['wait_duration'] = wait_duration
            
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
            
            # Get updated loading state
            updated_loading_state = await self._get_current_loading_state()
            self.test_state['spinner_visible_after'] = updated_loading_state.get('spinner_visible', False)
            self.test_state['results_visible_after'] = updated_loading_state.get('results_visible', False)
            self.test_state['loading_completed'] = updated_loading_state.get('loading_complete', False)
            
            # Evaluate the action
            action_type = action.get('action_type')
            
            if action_type == 9:  # WAIT action
                evaluation['valid'] = True
                evaluation['feedback'] = f'Wait executed - duration: {wait_duration}ms'
                self.test_state['wait_executed'] = True
                
                # Check if spinner disappeared
                if not self.test_state['spinner_visible_after']:
                    self.validator.spinner_disappeared = True
                    print(f"Spinner disappears and hotel results appear")
                
                # Check if results appeared
                if self.test_state['results_visible_after']:
                    self.validator.hotel_results_appeared = True
                    evaluation['feedback'] += ' - hotel results successfully loaded'
                else:
                    evaluation['feedback'] += ' - hotel results not yet visible'
                
                evaluation['test_completed'] = True
                evaluation['validation_result'] = self.validator.validate_wait_result()
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected WAIT (9) only'
                
        except Exception as e:
            evaluation['feedback'] = f'Action execution failed: {str(e)}'
            self.logger.error(f"Failed to execute action: {e}")
        
        return evaluation
    
    def _extract_wait_duration(self, action: Dict) -> int:
        """Extract wait duration from action."""
        if 'duration' in action:
            return action['duration']
        elif 'wait_time' in action:
            return action['wait_time']
        elif 'timeout' in action:
            return action['timeout']
        else:
            return 3000  # Default 3 seconds
    
    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's action (wait) in the browser backend."""
        action_type = action.get('action_type')
        
        if action_type == 9:  # WAIT action
            wait_duration = self._extract_wait_duration(action)
            wait_seconds = wait_duration / 1000.0
            
            # Execute wait by sleeping
            await asyncio.sleep(wait_seconds)
            
            result = {
                'success': True,
                'wait_duration': wait_duration,
                'wait_completed': True
            }
            
            self.logger.info(f"Wait executed for {wait_duration}ms")
            return result
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
            'wait_executed': self.test_state['wait_executed'],
            'wait_duration': self.test_state['wait_duration'],
            'spinner_visible_before': self.test_state['spinner_visible_before'],
            'spinner_visible_after': self.test_state['spinner_visible_after'],
            'results_visible_before': self.test_state['results_visible_before'],
            'results_visible_after': self.test_state['results_visible_after'],
            'loading_completed': self.test_state['loading_completed'],
            'test_completed': self.test_state['wait_executed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'booking_loading_wait',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'wait_duration': self.validator.wait_duration,
            'validation_summary': self.validator.validate_wait_result(),
            'element_configuration': self.booking_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_booking_loading_wait(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for Booking.com loading wait with real browser integration.
    """
    test_env = BookingWaitTestEnvironment(url, backend_url, config_path)
    
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
def test_booking_loading_wait_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_booking_loading_wait(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_booking_loading_wait_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> BookingWaitTestEnvironment:
    """Factory function to create a Booking.com loading wait test environment."""
    return BookingWaitTestEnvironment(url, backend_url, config_path)