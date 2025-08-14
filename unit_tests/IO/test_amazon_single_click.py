"""
Amazon Single Click Test Framework
Purpose: Test agent's ability to perform precise single click on Add to Cart button
Tests: CLICK action only
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


class AmazonSingleClickValidator:
    """Validator for Amazon single click operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.click_executed = False
        self.exact_coordinates = None
        self.cart_counter_incremented = False
        self.button_feedback_appeared = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_exact_click_coordinates(self, coordinates: Dict) -> None:
        """Print exact click coordinates."""
        self.exact_coordinates = coordinates
        self.click_executed = True
        print(f"Exact click coordinates: x={coordinates['x']}, y={coordinates['y']}")
        self.logger.info(f"Click executed at precise coordinates: {coordinates}")
    
    def validate_single_click_result(self) -> Dict[str, Any]:
        """Validate the single click action result."""
        validation_result = {
            'click_executed': self.click_executed,
            'exact_coordinates_logged': self.exact_coordinates is not None,
            'cart_counter_incremented': self.cart_counter_incremented,
            'button_feedback_appeared': self.button_feedback_appeared,
            'single_click_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if click was executed
        if not self.click_executed:
            validation_result['errors'].append("Click action was not executed")
        
        # Check if coordinates were logged
        if not self.exact_coordinates:
            validation_result['errors'].append("Exact click coordinates were not logged")
        
        # Check if cart counter incremented
        if not self.cart_counter_incremented:
            validation_result['warnings'].append("Cart counter did not increment")
        
        # Check if button feedback appeared
        if not self.button_feedback_appeared:
            validation_result['warnings'].append("Button feedback did not appear")
        
        # Determine if single click was successful
        validation_result['single_click_successful'] = (
            self.click_executed and 
            self.exact_coordinates is not None
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['single_click_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result


class AmazonSingleClickTestEnvironment:
    """Test environment for Amazon single click with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = AmazonSingleClickValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 10)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.amazon_elements_config = self.config_loader.get_section('amazon_single_click_elements')
        
        self.test_state = {
            'current_step': 0,
            'click_executed': False,
            'add_to_cart_clicked': False,
            'cart_count_before': 0,
            'cart_count_after': 0,
            'button_feedback_visible': False,
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
        """Get initial test setup with Echo Dot product page and visible Add to Cart button."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show Echo Dot product page)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect Add to Cart button and cart counter
            element_detection = await self._detect_cart_elements()
            cart_state = await self._get_current_cart_state()
            
            # Store initial cart count
            self.test_state['cart_count_before'] = cart_state.get('cart_count', 0)
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_product_page',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Execute single precise click on Add to Cart button',
                'element_detection': element_detection,
                'cart_state': cart_state,
                'initial_screenshot': screenshot_data,
                'cart_elements': self.amazon_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_action': 'CLICK(add_to_cart_button) once only',
                    'expected_result': 'cart counter increments and button feedback appears'
                },
                'history': [
                    'Amazon Echo Dot product page loaded',
                    'Add to Cart button clearly visible',
                    'Product details and pricing displayed',
                    'Ready for single click action'
                ],
                'current_cart': {
                    'count': self.test_state['cart_count_before'],
                    'add_to_cart_visible': element_detection.get('add_to_cart_button', {}).get('found', False)
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Amazon single click test initialized: {self.url}")
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
    
    async def _detect_cart_elements(self) -> Dict:
        """Detect Add to Cart button and cart counter elements."""
        detected_elements = {}
        
        for element_type, config in self.amazon_elements_config.items():
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
                                button_text: element.value || element.textContent || '',
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
    
    async def _get_current_cart_state(self) -> Dict:
        """Get current cart count and button state."""
        try:
            eval_script = """
            () => {
                const result = {
                    cart_count: 0,
                    add_to_cart_enabled: false,
                    button_feedback_visible: false
                };
                
                // Get cart count
                const cartCounter = document.querySelector('#nav-cart-count') ||
                                  document.querySelector('.cart-count') ||
                                  document.querySelector('[data-testid="cart-counter"]');
                
                if (cartCounter) {
                    const countText = cartCounter.textContent || cartCounter.innerText || '0';
                    result.cart_count = parseInt(countText) || 0;
                }
                
                // Check Add to Cart button state
                const addToCartButton = document.querySelector('#add-to-cart-button') ||
                                      document.querySelector('[data-testid="add-to-cart"]') ||
                                      document.querySelector('input[name="submit.add-to-cart"]');
                
                if (addToCartButton) {
                    result.add_to_cart_enabled = !addToCartButton.disabled;
                }
                
                // Check for button feedback (success message, animation, etc.)
                const feedbackElements = [
                    document.querySelector('.add-to-cart-success'),
                    document.querySelector('[data-testid="cart-feedback"]'),
                    document.querySelector('.cart-notification'),
                    document.querySelector('.added-to-cart')
                ];
                
                for (const element of feedbackElements) {
                    if (element) {
                        const rect = element.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            result.button_feedback_visible = true;
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
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to get cart state: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's single click action."""
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
            
            # Log exact click coordinates
            if 'coordinates' in action:
                self.validator.log_exact_click_coordinates(action['coordinates'])
            elif browser_result and 'click_coordinates' in browser_result:
                self.validator.log_exact_click_coordinates(browser_result['click_coordinates'])
            
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
            
            # Get updated cart state
            await asyncio.sleep(1)  # Wait for cart update
            updated_cart_state = await self._get_current_cart_state()
            self.test_state['cart_count_after'] = updated_cart_state.get('cart_count', 0)
            
            # Evaluate the action
            action_type = action.get('action_type')
            element_name = action.get('element_name', '').lower()
            
            if action_type == 6:  # CLICK action
                if 'add' in element_name and 'cart' in element_name:
                    # Add to Cart button clicked
                    evaluation['valid'] = True
                    evaluation['feedback'] = 'Add to Cart button clicked successfully'
                    self.test_state['click_executed'] = True
                    self.test_state['add_to_cart_clicked'] = True
                    
                    # Check if cart counter incremented
                    if self.test_state['cart_count_after'] > self.test_state['cart_count_before']:
                        self.validator.cart_counter_incremented = True
                        print(f"Cart counter incremented: {self.test_state['cart_count_before']} → {self.test_state['cart_count_after']}")
                    
                    # Check if button feedback appeared
                    if updated_cart_state.get('button_feedback_visible'):
                        self.validator.button_feedback_appeared = True
                        self.test_state['button_feedback_visible'] = True
                        print("Button feedback appears: Add to Cart success indication visible")
                    
                    evaluation['test_completed'] = True
                    evaluation['validation_result'] = self.validator.validate_single_click_result()
                else:
                    evaluation['feedback'] = f'Clicked wrong element - expected Add to Cart button, got: {element_name}'
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected CLICK (6) only'
                
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
                if 'add' in element_name and 'cart' in element_name:
                    coords = self.amazon_elements_config.get('add_to_cart_button', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.8), coords.get('y', 0.5)
                else:
                    x, y = 0.5, 0.5  # Default center click
            else:
                x, y = 0.5, 0.5  # Default center click
            
            # Execute click via browser backend
            payload = {"x": x, "y": y}
            
            try:
                async with self.http_session.post(f"{self.backend_url}/click", json=payload) as response:
                    result = await response.json()
                    result['click_coordinates'] = {'x': x, 'y': y}  # Add coordinates to result
                    self.logger.info(f"Click executed at ({x}, {y}): {result}")
                    return result
            except Exception as e:
                self.logger.error(f"Click execution failed: {e}")
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
            'click_executed': self.test_state['click_executed'],
            'add_to_cart_clicked': self.test_state['add_to_cart_clicked'],
            'cart_count_before': self.test_state['cart_count_before'],
            'cart_count_after': self.test_state['cart_count_after'],
            'button_feedback_visible': self.test_state['button_feedback_visible'],
            'test_completed': self.test_state['click_executed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'amazon_single_click',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'click_coordinates': self.validator.exact_coordinates,
            'validation_summary': self.validator.validate_single_click_result(),
            'element_configuration': self.amazon_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_amazon_single_click(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for Amazon single click with real browser integration.
    """
    test_env = AmazonSingleClickTestEnvironment(url, backend_url, config_path)
    
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
def test_amazon_single_click_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_amazon_single_click(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_amazon_single_click_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> AmazonSingleClickTestEnvironment:
    """Factory function to create an Amazon single click test environment."""
    return AmazonSingleClickTestEnvironment(url, backend_url, config_path)