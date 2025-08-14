"""
Booking.com Room Capacity Logic Test Framework
Purpose: Test agent's ability to reduce adult count to resolve room capacity issues
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
from browser_env.actions import create_click_action, ActionTypes

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


class BookingRoomCapacityValidator:
    """Validator for booking room capacity operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.occupancy_changes = []
        self.adult_decrements = 0
        self.capacity_warning_cleared = False
        self.room_bookable = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_occupancy_change(self, change_type: str, occupancy_data: Dict) -> None:
        """Log occupancy changes with capacity validation."""
        change_log = {
            'change_id': len(self.occupancy_changes) + 1,
            'timestamp': datetime.now().isoformat(),
            'change_type': change_type,
            'occupancy_data': occupancy_data
        }
        
        self.occupancy_changes.append(change_log)
        self.logger.info(f"Occupancy change: {change_type}")
        
        if change_type == 'ADULT_DECREMENT':
            self.adult_decrements += 1
            print(f"Occupancy changes: Adults reduced to {occupancy_data.get('adult_count', 'unknown')}")
        
        # Check capacity validation
        if occupancy_data.get('capacity_valid'):
            self.capacity_warning_cleared = True
            self.room_bookable = True
            print("Capacity validation: Room becomes bookable (capacity ≤ max)")
    
    def validate_capacity_logic_result(self) -> Dict[str, Any]:
        """Validate the complete room capacity logic sequence."""
        validation_result = {
            'adult_decrements_executed': self.adult_decrements >= 2,
            'capacity_warning_cleared': self.capacity_warning_cleared,
            'room_bookable': self.room_bookable,
            'total_decrements': self.adult_decrements,
            'expected_decrements': 2,
            'errors': [],
            'warnings': []
        }
        
        # Check if correct number of decrements were executed
        if self.adult_decrements < 2:
            validation_result['errors'].append(f"Need 2 adult decrements, got {self.adult_decrements}")
        elif self.adult_decrements > 2:
            validation_result['warnings'].append(f"More decrements than needed: {self.adult_decrements}")
        
        # Check if capacity warning was cleared
        if not self.capacity_warning_cleared:
            validation_result['errors'].append("Capacity warning was not cleared")
        
        # Check if room became bookable
        if not self.room_bookable:
            validation_result['errors'].append("Room did not become bookable")
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['adult_decrements_executed'] and
            validation_result['capacity_warning_cleared'] and
            validation_result['room_bookable'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    def get_occupancy_log(self) -> List[Dict]:
        """Return the complete log of occupancy changes."""
        return self.occupancy_changes


class BookingRoomCapacityTestEnvironment:
    """Test environment for booking room capacity logic with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = BookingRoomCapacityValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 30)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.booking_elements_config = self.config_loader.get_section('booking_room_capacity_elements')
        
        self.test_state = {
            'current_step': 0,
            'occupancy_selector_open': False,
            'adult_count': 4,  # Starting with 4 adults
            'capacity_warning_visible': True,
            'room_bookable': False,
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
        """Get initial test setup with room capacity warning visible."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show room with capacity issue)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect occupancy elements and capacity warning
            element_detection = await self._detect_occupancy_elements()
            capacity_warning = await self._check_capacity_warning()
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_capacity_warning',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Reduce adult count to resolve room capacity warning and make room bookable',
                'element_detection': element_detection,
                'capacity_warning': capacity_warning,
                'initial_screenshot': screenshot_data,
                'occupancy_elements': self.booking_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_sequence': ['CLICK(adult_decrement)', 'CLICK(adult_decrement)'],
                    'expected_final_state': 'capacity warning cleared, room bookable'
                },
                'history': [
                    'Hotel property page loaded',
                    'Occupancy selector shows "2 adults, 1 child"',
                    'Currently 4 adults selected (over capacity)',
                    'Room capacity warning is visible',
                    'Room is not bookable due to capacity issue'
                ],
                'current_occupancy': {
                    'adults': self.test_state['adult_count'],
                    'children': 1,
                    'capacity_exceeded': True
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.test_state['occupancy_selector_open'] = element_detection.get('adult_decrement', {}).get('found', False)
            self.test_state['capacity_warning_visible'] = capacity_warning.get('found', False)
            
            self.logger.info(f"Booking room capacity test initialized: {self.url}")
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
    
    async def _detect_occupancy_elements(self) -> Dict:
        """Detect occupancy selector elements on the current page."""
        detected_elements = {}
        
        for element_type, config in self.booking_elements_config.items():
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
                                enabled: !element.disabled,
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
                                    'coordinates': result.get('coordinates'),
                                    'element_info': result
                                }
                                break
                        
                except Exception as e:
                    self.logger.warning(f"Failed to check selector {selector}: {e}")
                    continue
            
            detected_elements[element_type] = detected
        
        return detected_elements
    
    async def _check_capacity_warning(self) -> Dict:
        """Check if capacity warning is visible."""
        warning_config = self.booking_elements_config.get('capacity_warning', {})
        
        for selector in warning_config.get('selectors', []):
            try:
                eval_script = f"""
                () => {{
                    const element = document.querySelector('{selector}');
                    if (element) {{
                        const rect = element.getBoundingClientRect();
                        return {{
                            found: true,
                            visible: rect.width > 0 && rect.height > 0,
                            text_content: element.textContent || '',
                            display_style: window.getComputedStyle(element).display
                        }};
                    }}
                    return {{ found: false }};
                }}
                """
                
                payload = {"script": eval_script}
                async with self.http_session.post(f"{self.backend_url}/evaluate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('found') and result.get('visible'):
                            return result
                    
            except Exception as e:
                self.logger.warning(f"Failed to check warning selector {selector}: {e}")
                continue
        
        return {'found': False}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's room capacity action."""
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
            
            # Get current occupancy state
            occupancy_state = await self._get_current_occupancy_state()
            
            # Evaluate the action
            action_type = action.get('action_type')
            element_name = action.get('element_name', '').lower()
            
            if action_type == 6:  # CLICK action
                if 'decrement' in element_name or 'minus' in element_name:
                    if occupancy_state.get('adult_count', 0) < self.test_state['adult_count']:
                        evaluation['valid'] = True
                        evaluation['feedback'] = f'Adult decrement executed - count reduced to {occupancy_state["adult_count"]}'
                        
                        # Update internal state
                        self.test_state['adult_count'] = occupancy_state['adult_count']
                        
                        # Log the occupancy change
                        self.validator.log_occupancy_change('ADULT_DECREMENT', {
                            'adult_count': occupancy_state['adult_count'],
                            'capacity_valid': occupancy_state.get('capacity_valid', False),
                            'warning_cleared': not occupancy_state.get('warning_visible', True)
                        })
                        
                        # Check if capacity is now valid (2 adults max)
                        if occupancy_state['adult_count'] <= 2:
                            self.test_state['capacity_warning_visible'] = False
                            self.test_state['room_bookable'] = True
                            evaluation['test_completed'] = True
                            evaluation['validation_result'] = self.validator.validate_capacity_logic_result()
                        else:
                            evaluation['expected_next'] = 'Continue clicking adult decrement to reach 2 adults maximum'
                    else:
                        evaluation['feedback'] = 'Adult decrement clicked but count did not decrease'
                else:
                    evaluation['feedback'] = f'Clicked wrong element - expected adult decrement button, got: {element_name}'
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected CLICK (6)'
                
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
                if 'decrement' in element_name or 'minus' in element_name:
                    coords = self.booking_elements_config.get('adult_decrement', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.25), coords.get('y', 0.5)
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
        else:
            raise Exception(f"Unsupported action type for this test: {action_type}")
    
    async def _get_current_occupancy_state(self) -> Dict:
        """Get the current occupancy state and capacity validation."""
        try:
            eval_script = """
            () => {
                const result = {
                    adult_count: null,
                    child_count: null,
                    capacity_valid: false,
                    warning_visible: false,
                    room_bookable: false
                };
                
                // Try to find adult count display
                const adultCountElements = [
                    document.querySelector('[data-testid="adults-count"]'),
                    document.querySelector('.adults-count'),
                    document.querySelector('input[name*="adults"]')
                ];
                
                for (const element of adultCountElements) {
                    if (element) {
                        const count = parseInt(element.value || element.textContent || '0');
                        if (!isNaN(count)) {
                            result.adult_count = count;
                            break;
                        }
                    }
                }
                
                // Check capacity warning visibility
                const warningElements = [
                    document.querySelector('.capacity-warning'),
                    document.querySelector('[role="alert"]'),
                    document.querySelector('.occupancy-error')
                ];
                
                for (const element of warningElements) {
                    if (element) {
                        const rect = element.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            result.warning_visible = true;
                            break;
                        }
                    }
                }
                
                // Determine capacity validity (2 adults max)
                if (result.adult_count !== null && result.adult_count <= 2) {
                    result.capacity_valid = true;
                    result.room_bookable = !result.warning_visible;
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
            self.logger.error(f"Failed to get occupancy state: {e}")
            return {}
    
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
            'adult_count': self.test_state['adult_count'],
            'capacity_warning_visible': self.test_state['capacity_warning_visible'],
            'room_bookable': self.test_state['room_bookable'],
            'test_completed': self.test_state['room_bookable'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'booking_room_capacity_logic',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'occupancy_changes': self.validator.get_occupancy_log(),
            'validation_summary': self.validator.validate_capacity_logic_result(),
            'element_configuration': self.booking_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_booking_room_capacity_logic(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for booking room capacity logic with real browser integration.
    
    Args:
        url: URL of the booking property page to test
        run_agent_func: External function that takes initial_observation and returns list of actions
        backend_url: Browser-in-browser backend URL (optional, uses config)
        config_path: Path to configuration file (optional)
    
    Returns:
        Complete test results dictionary
    """
    test_env = BookingRoomCapacityTestEnvironment(url, backend_url, config_path)
    
    try:
        # Provide initial observation to the external agent
        initial_obs = await test_env.get_initial_observation()
        
        if not initial_obs.get('success'):
            return {
                'success': False,
                'error': 'Failed to initialize browser environment',
                'details': initial_obs,
                'timestamp': datetime.now().isoformat()
            }
        
        # Let the external agent decide what actions to take
        agent_actions = run_agent_func(initial_obs)
        
        # Validate that agent returned a list of actions
        if not isinstance(agent_actions, list):
            await test_env.cleanup()
            return {
                'success': False,
                'error': 'Agent must return a list of actions',
                'timestamp': datetime.now().isoformat()
            }
        
        # Evaluate each action
        evaluations = []
        test_completed = False
        
        for action in agent_actions:
            evaluation = await test_env.evaluate_agent_action(action)
            evaluations.append(evaluation)
            
            if evaluation['test_completed']:
                test_completed = True
                break
        
        # Get final test status and generate report
        test_status = test_env.get_test_status()
        test_report = test_env.get_test_report()
        
        # Cleanup browser session
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
def test_booking_room_capacity_logic_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_booking_room_capacity_logic(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_booking_room_capacity_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> BookingRoomCapacityTestEnvironment:
    """Factory function to create a booking room capacity test environment."""
    return BookingRoomCapacityTestEnvironment(url, backend_url, config_path)