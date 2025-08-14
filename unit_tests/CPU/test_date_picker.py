"""
Date Picker Plugin Framework with Browser-in-Browser Integration
Purpose: Modular components for date picker testing that integrates with browser-in-browser backend
"""
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import base64
import io

# Browser integration imports
import sys
sys.path.append('/home/ubuntu/webarena')
from browser_env.actions import create_click_action, create_mouse_click_action, ActionTypes


class ConfigLoader:
    """Loads and manages configuration from config.json file."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config.json in the same directory as this file
            config_path = Path(__file__).parent / "config.json"
        
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
            logging.getLogger(__name__).info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logging.getLogger(__name__).warning(f"Config file not found: {self.config_path}, using defaults")
            self._config = self._get_default_config()
        except json.JSONDecodeError as e:
            logging.getLogger(__name__).error(f"Invalid JSON in config file: {e}, using defaults")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not available."""
        return {
            "test_configuration": {
                "max_steps": 3,
                "timeout_seconds": 30,
                "screenshot_enabled": True,
                "detailed_logging": True
            },
            "date_picker_elements": {
                "check_in": {
                    "selectors": [
                        "[data-testid='checkin-date']",
                        "[aria-label*='Check-in']",
                        "input[placeholder*='Check-in']",
                        ".checkin-date",
                        "#checkin_date"
                    ],
                    "coordinates": {"x": 0.3, "y": 0.4},
                    "description": "Check-in date picker input"
                },
                "check_out": {
                    "selectors": [
                        "[data-testid='checkout-date']",
                        "[aria-label*='Check-out']", 
                        "input[placeholder*='Check-out']",
                        ".checkout-date",
                        "#checkout_date"
                    ],
                    "coordinates": {"x": 0.7, "y": 0.4},
                    "description": "Check-out date picker input"
                }
            },
            "validation_rules": {
                "both_dates_required": True,
                "check_in_before_check_out": True,
                "no_past_dates": True,
                "max_booking_window_days": 365
            },
            "browser_settings": {
                "backend_url": "http://localhost:8000",
                "screenshot_frequency": "per_action",
                "wait_for_load": True,
                "network_idle_timeout": 2000
            },
            "logging": {
                "level": "INFO",
                "file_path": "/home/ubuntu/webarena/unit_tests/CPU/date_picker_test.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'test_configuration.max_steps')."""
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    @property
    def config(self) -> Dict:
        """Get the full configuration dictionary."""
        return self._config.copy()


class DatePickerTestEnvironment:
    """Test environment for evaluating agent date picker interactions with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = DatePickerValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = self.config_loader.get('test_configuration.max_steps', 3)
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 30)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.date_elements_config = self.config_loader.get_section('date_picker_elements')
        
        # Browser settings from config
        self.screenshot_frequency = self.config_loader.get('browser_settings.screenshot_frequency', 'per_action')
        self.wait_for_load = self.config_loader.get('browser_settings.wait_for_load', True)
        self.network_idle_timeout = self.config_loader.get('browser_settings.network_idle_timeout', 2000)
        
        self.test_state = {
            'current_step': 0,
            'check_in_selected': False,
            'check_out_selected': False,
            'action_history': [],
            'screenshots': [],
            'browser_session_active': False
        }
        
        # HTTP client for browser backend communication
        self.http_session = None
        
        # Configure logging based on config
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_level = self.config_loader.get('logging.level', 'INFO')
        log_format = self.config_loader.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = self.config_loader.get('logging.file_path')
        
        # Configure logger
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")
    
    async def get_initial_observation(self) -> Dict:
        """Get the initial test setup and provide context to the agent with real browser state."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect date picker elements on the actual page
            detected_elements = await self._detect_date_picker_elements()
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Select check-in and check-out dates from the dual calendar picker',
                'expected_elements': self.date_elements_config,
                'detected_elements': detected_elements,
                'initial_screenshot': screenshot_data,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_sequence': ['check_in_selection', 'check_out_selection'],
                    'date_order_constraint': 'check_out > check_in'
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Browser session initialized for: {self.url}")
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
        
        # Check if browser backend is running
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
                    # Convert to base64 for storage/transmission
                    screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                    return screenshot_b64
                else:
                    raise Exception(f"Screenshot request failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to get screenshot: {e}")
            raise
    
    async def _detect_date_picker_elements(self) -> Dict:
        """Detect actual date picker elements on the current page."""
        detected_elements = {}
        
        for element_type, config in self.date_elements_config.items():
            detected = {
                'found': False,
                'selector_used': None,
                'coordinates': None,
                'element_info': None
            }
            
            # Try each selector until we find the element
            for selector in config['selectors']:
                try:
                    # Use browser backend to check if element exists
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
                                tag: element.tagName,
                                type: element.type || '',
                                placeholder: element.placeholder || '',
                                value: element.value || ''
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
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """
        Execute and evaluate an agent's action in the real browser environment.
        
        Args:
            action: The action taken by the agent
            
        Returns:
            Evaluation result including feedback, browser response, and validation
        """
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
            # Execute the action in the browser
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
            
            # Extract current form values
            form_values = await self._extract_current_date_values()
            
            # Evaluate based on action type and results
            action_type = action.get('action_type')
            element_name = action.get('element_name', '').lower()
            
            if action_type == 6:  # CLICK action
                if 'check' in element_name and 'in' in element_name:
                    # Check-in date selection
                    if not self.test_state['check_in_selected']:
                        if form_values.get('check_in'):
                            evaluation['valid'] = True
                            evaluation['feedback'] = 'Check-in date selection successful'
                            evaluation['expected_next'] = 'Select check-out date (must be after check-in)'
                            self.test_state['check_in_selected'] = True
                            
                            # Log the actual selected date
                            self.validator.log_date_selection('check_in', form_values['check_in'], 
                                                            self.date_elements_config['check_in'])
                        else:
                            evaluation['feedback'] = 'Check-in click executed but no date selected'
                    else:
                        evaluation['feedback'] = 'Check-in date already selected'
                        
                elif 'check' in element_name and 'out' in element_name:
                    # Check-out date selection
                    if self.test_state['check_in_selected'] and not self.test_state['check_out_selected']:
                        if form_values.get('check_out'):
                            evaluation['valid'] = True
                            evaluation['feedback'] = 'Check-out date selection successful'
                            self.test_state['check_out_selected'] = True
                            
                            # Log the actual selected date
                            self.validator.log_date_selection('check_out', form_values['check_out'],
                                                            self.date_elements_config['check_out'])
                            
                            # Test is now complete - validate
                            evaluation['test_completed'] = True
                            evaluation['validation_result'] = self.validator.validate_date_fields()
                        else:
                            evaluation['feedback'] = 'Check-out click executed but no date selected'
                            
                    elif not self.test_state['check_in_selected']:
                        evaluation['feedback'] = 'Must select check-in date first'
                    else:
                        evaluation['feedback'] = 'Check-out date already selected'
                else:
                    evaluation['feedback'] = 'Clicked but target element unclear - should target date picker'
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
                # Try to find element based on name
                element_name = action['element_name'].lower()
                if 'check' in element_name and 'in' in element_name:
                    coords = self.date_elements_config['check_in']['coordinates']
                    x, y = coords['x'], coords['y']
                elif 'check' in element_name and 'out' in element_name:
                    coords = self.date_elements_config['check_out']['coordinates']
                    x, y = coords['x'], coords['y']
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
            payload = {"key": text}
            
            try:
                async with self.http_session.post(f"{self.backend_url}/keyboard", json=payload) as response:
                    result = await response.json()
                    self.logger.info(f"Type executed: {text}")
                    return result
            except Exception as e:
                self.logger.error(f"Type execution failed: {e}")
                raise
        else:
            raise Exception(f"Unsupported action type: {action_type}")
    
    async def _extract_current_date_values(self) -> Dict:
        """Extract current date values from the form inputs."""
        extract_script = """
        () => {
            const result = {};
            
            // Common selectors for date inputs
            const selectors = [
                '[data-testid="checkin-date"]',
                '[aria-label*="Check-in"]',
                'input[placeholder*="Check-in"]',
                '.checkin-date',
                '#checkin_date'
            ];
            
            // Try to find check-in input
            for (const selector of selectors) {
                const element = document.querySelector(selector);
                if (element && element.value) {
                    result.check_in = element.value;
                    break;
                }
            }
            
            const checkoutSelectors = [
                '[data-testid="checkout-date"]',
                '[aria-label*="Check-out"]',
                'input[placeholder*="Check-out"]',
                '.checkout-date',
                '#checkout_date'
            ];
            
            // Try to find check-out input
            for (const selector of checkoutSelectors) {
                const element = document.querySelector(selector);
                if (element && element.value) {
                    result.check_out = element.value;
                    break;
                }
            }
            
            return result;
        }
        """
        
        try:
            payload = {"script": extract_script}
            async with self.http_session.post(f"{self.backend_url}/evaluate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.info(f"Extracted date values: {result}")
                    return result
                else:
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to extract date values: {e}")
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
            'check_in_selected': self.test_state['check_in_selected'],
            'check_out_selected': self.test_state['check_out_selected'],
            'test_completed': self.test_state['check_in_selected'] and self.test_state['check_out_selected'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'plugin_version': '1.0.0',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'date_selections': self.validator.get_selection_log(),
            'validation_summary': self.validator.validate_date_fields(),
            'element_configuration': self.date_elements_config,
            'timestamp': datetime.now().isoformat()
        }


class DatePickerValidator:
    """Enhanced validator for date picker operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.date_selections = []
        self.check_in_date = None
        self.check_out_date = None
        
        # Use provided config loader or create default one
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
        
        # Load validation rules from config
        self.validation_rules = self.config_loader.get_section('validation_rules')
        if not self.validation_rules:
            self.validation_rules = self._get_default_validation_rules()
    
    def _get_default_validation_rules(self) -> Dict:
        """Get default validation rules if config is not available."""
        return {
            'both_dates_required': True,
            'check_in_before_check_out': True,
            'no_past_dates': True,
            'max_booking_window_days': 365,
            'min_stay_duration_days': 1,
            'max_stay_duration_days': 30
        }
    
    def log_date_selection(self, date_type: str, date_value: str, element_info: Dict) -> None:
        """Log date selection with enhanced details."""
        selection = {
            'selection_id': len(self.date_selections) + 1,
            'timestamp': datetime.now().isoformat(),
            'date_type': date_type,
            'date_value': date_value,
            'element_info': element_info,
            'validation_at_selection': self._validate_single_date(date_value)
        }
        
        self.date_selections.append(selection)
        self.logger.info(f"Date selected: {date_type} = {date_value}")
        
        # Update internal state
        if date_type == 'check_in':
            self.check_in_date = date_value
        elif date_type == 'check_out':
            self.check_out_date = date_value
    
    def _validate_single_date(self, date_value: str) -> Dict:
        """Validate a single date value."""
        validation = {'valid': True, 'issues': []}
        
        try:
            date_obj = datetime.strptime(date_value, '%Y-%m-%d')
            
            # Check if date is in the past
            if self.validation_rules['no_past_dates'] and date_obj.date() < datetime.now().date():
                validation['valid'] = False
                validation['issues'].append('Date is in the past')
            
            # Check booking window
            days_from_now = (date_obj.date() - datetime.now().date()).days
            if days_from_now > self.validation_rules['max_booking_window_days']:
                validation['valid'] = False
                validation['issues'].append(f'Date exceeds maximum booking window ({self.validation_rules["max_booking_window_days"]} days)')
                
        except ValueError:
            validation['valid'] = False
            validation['issues'].append('Invalid date format')
        
        return validation
    
    def validate_date_fields(self) -> Dict[str, Any]:
        """Comprehensive validation of both date fields."""
        validation_result = {
            'both_populated': False,
            'date_order_valid': False,
            'individual_dates_valid': True,
            'check_in_date': self.check_in_date,
            'check_out_date': self.check_out_date,
            'errors': [],
            'warnings': [],
            'validation_details': {}
        }
        
        # Check if both dates are populated
        if self.check_in_date and self.check_out_date:
            validation_result['both_populated'] = True
        else:
            validation_result['errors'].append("Both check-in and check-out dates must be selected")
        
        # Validate individual dates
        if self.check_in_date:
            check_in_validation = self._validate_single_date(self.check_in_date)
            validation_result['validation_details']['check_in'] = check_in_validation
            if not check_in_validation['valid']:
                validation_result['individual_dates_valid'] = False
                validation_result['errors'].extend([f"Check-in: {issue}" for issue in check_in_validation['issues']])
        
        if self.check_out_date:
            check_out_validation = self._validate_single_date(self.check_out_date)
            validation_result['validation_details']['check_out'] = check_out_validation
            if not check_out_validation['valid']:
                validation_result['individual_dates_valid'] = False
                validation_result['errors'].extend([f"Check-out: {issue}" for issue in check_out_validation['issues']])
        
        # Check date order
        if self.check_in_date and self.check_out_date:
            try:
                check_in = datetime.strptime(self.check_in_date, '%Y-%m-%d')
                check_out = datetime.strptime(self.check_out_date, '%Y-%m-%d')
                
                if check_in < check_out:
                    validation_result['date_order_valid'] = True
                elif check_in == check_out:
                    validation_result['errors'].append("Check-in and check-out dates cannot be the same")
                else:
                    validation_result['errors'].append("Check-in date must be before check-out date")
                    
                # Calculate stay duration
                duration = (check_out - check_in).days
                validation_result['stay_duration_days'] = duration
                
                # Check against configured limits
                min_duration = self.validation_rules.get('min_stay_duration_days', 1)
                max_duration = self.validation_rules.get('max_stay_duration_days', 30)
                
                if duration < min_duration:
                    validation_result['errors'].append(f"Stay duration too short: {duration} days (minimum: {min_duration})")
                elif duration > max_duration:
                    validation_result['warnings'].append(f"Long stay duration: {duration} days (maximum recommended: {max_duration})")
                    
            except ValueError as e:
                validation_result['errors'].append(f"Date parsing error: {e}")
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['both_populated'] and 
            validation_result['date_order_valid'] and 
            validation_result['individual_dates_valid'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    def get_selection_log(self) -> List[Dict]:
        """Return the complete log of date selections."""
        return self.date_selections


# Convenience function for quick integration
def create_date_picker_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> DatePickerTestEnvironment:
    """
    Factory function to create a date picker test environment.
    
    Args:
        url: The URL to test
        backend_url: Browser-in-browser backend URL (optional, uses config)
        config_path: Path to configuration file (optional)
    
    Returns:
        Configured DatePickerTestEnvironment instance
    """
    return DatePickerTestEnvironment(url, backend_url, config_path)


# Test runner function for external agent systems
async def test_booking_date_picker(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Standalone test function that evaluates an external agent's date picker interaction with real browser.
    
    This function provides a test environment to an external agent and evaluates
    its performance on the date picker task using browser-in-browser integration.
    
    Args:
        url: URL of the booking page to test
        run_agent_func: External function that takes initial_observation and returns list of actions
                       Signature: run_agent_func(initial_observation: Dict) -> List[Dict]
        backend_url: Browser-in-browser backend URL (optional, uses config if not provided)
        config_path: Path to configuration file (optional, uses default config.json)
    
    Returns:
        Complete test results dictionary with success/failure and detailed evaluation
    """
    test_env = DatePickerTestEnvironment(url, backend_url, config_path)
    
    try:
        # Provide initial observation to the external agent (real browser state)
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
        
        # Evaluate each action the external agent took (with real browser execution)
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
def test_booking_date_picker_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Synchronous wrapper for the async test function.
    
    Args:
        url: URL of the booking page to test
        run_agent_func: External agent function
        backend_url: Browser-in-browser backend URL (optional, uses config)
        config_path: Path to configuration file (optional)
    
    Returns:
        Test results dictionary
    """
    return asyncio.run(test_booking_date_picker(url, run_agent_func, backend_url, config_path))