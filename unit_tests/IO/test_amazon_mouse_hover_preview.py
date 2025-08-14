"""
Amazon Mouse Hover Preview Test Framework
Purpose: Test agent's ability to perform mouse hover to show product preview
Tests: MOVE MOUSE action only
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
from browser_env.actions import create_move_mouse_action, ActionTypes

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


class AmazonMouseHoverValidator:
    """Validator for Amazon mouse hover operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.mouse_move_executed = False
        self.hover_coordinates = None
        self.product_preview_appeared = False
        self.quick_view_shown = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_mouse_hover_coordinates(self, coordinates: Dict) -> None:
        """Print mouse hover coordinates."""
        self.hover_coordinates = coordinates
        self.mouse_move_executed = True
        print(f"Mouse hover coordinates: ({coordinates['x']}, {coordinates['y']})")
        self.logger.info(f"Mouse move executed to coordinates: {coordinates}")
    
    def validate_mouse_hover_result(self) -> Dict[str, Any]:
        """Validate the mouse hover action result."""
        validation_result = {
            'mouse_move_executed': self.mouse_move_executed,
            'coordinates_logged': self.hover_coordinates is not None,
            'product_preview_appeared': self.product_preview_appeared,
            'quick_view_shown': self.quick_view_shown,
            'hover_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if mouse move was executed
        if not self.mouse_move_executed:
            validation_result['errors'].append("Mouse move action was not executed")
        
        # Check if coordinates were logged
        if not self.hover_coordinates:
            validation_result['errors'].append("Mouse hover coordinates were not logged")
        
        # Check if product preview appeared
        if not self.product_preview_appeared:
            validation_result['errors'].append("Product preview did not appear on hover")
        
        # Check if quick view was shown
        if not self.quick_view_shown:
            validation_result['warnings'].append("Quick view popup not detected")
        
        # Determine if hover was successful
        validation_result['hover_successful'] = (
            self.mouse_move_executed and 
            self.product_preview_appeared
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['hover_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result


class AmazonMouseHoverTestEnvironment:
    """Test environment for Amazon mouse hover with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = AmazonMouseHoverValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 10)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.amazon_elements_config = self.config_loader.get_section('amazon_hover_elements')
        
        self.test_state = {
            'current_step': 0,
            'mouse_move_executed': False,
            'hover_coordinates': None,
            'preview_visible_before': False,
            'preview_visible_after': False,
            'quick_view_popup_visible': False,
            'product_cards_found': False,
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
        """Get initial test setup with product grid ready for hover."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show Amazon product grid)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect product cards and preview elements
            product_detection = await self._detect_product_elements()
            preview_state = await self._get_current_preview_state()
            
            # Store initial state
            self.test_state['product_cards_found'] = product_detection.get('product_card', {}).get('found', False)
            self.test_state['preview_visible_before'] = preview_state.get('preview_visible', False)
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_product_grid',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Execute mouse hover on product card to show preview',
                'product_detection': product_detection,
                'preview_state': preview_state,
                'initial_screenshot': screenshot_data,
                'hover_elements': self.amazon_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_action': 'MOVE_MOUSE(product_card_coordinates) once only',
                    'expected_result': 'product preview overlay appears on hover'
                },
                'history': [
                    'Amazon product grid page loaded',
                    'Product cards are displayed in grid layout',
                    'Product images and prices visible',
                    'Hover should trigger preview overlay',
                    'Ready for mouse hover action'
                ],
                'current_product_grid': {
                    'product_cards_found': self.test_state['product_cards_found'],
                    'preview_visible': self.test_state['preview_visible_before'],
                    'hover_target_ready': True
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Amazon mouse hover test initialized: {self.url}")
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
    
    async def _detect_product_elements(self) -> Dict:
        """Detect product cards and preview elements."""
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
                                title: element.title || element.getAttribute('alt') || '',
                                data_asin: element.getAttribute('data-asin') || '',
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
    
    async def _get_current_preview_state(self) -> Dict:
        """Get current product preview state."""
        try:
            eval_script = """
            () => {
                const result = {
                    preview_visible: false,
                    quick_view_visible: false,
                    preview_content_count: 0
                };
                
                // Check for product preview overlays
                const previewElements = [
                    document.querySelector('.product-preview'),
                    document.querySelector('[data-testid="product-overlay"]'),
                    document.querySelector('.quick-view'),
                    document.querySelector('.hover-preview'),
                    document.querySelector('.product-popup')
                ];
                
                for (const preview of previewElements) {
                    if (preview) {
                        const rect = preview.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            result.preview_visible = true;
                            result.preview_content_count++;
                            
                            // Check if it's a quick view specifically
                            const text = preview.textContent || preview.className || '';
                            if (text.toLowerCase().includes('quick') || text.toLowerCase().includes('view')) {
                                result.quick_view_visible = true;
                            }
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
            self.logger.error(f"Failed to get preview state: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's mouse hover action."""
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
            
            # Log mouse hover coordinates
            if 'coordinates' in action:
                self.validator.log_mouse_hover_coordinates(action['coordinates'])
                self.test_state['hover_coordinates'] = action['coordinates']
            elif browser_result and 'hover_coordinates' in browser_result:
                self.validator.log_mouse_hover_coordinates(browser_result['hover_coordinates'])
                self.test_state['hover_coordinates'] = browser_result['hover_coordinates']
            
            # Wait for hover effects to trigger
            await asyncio.sleep(1)
            
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
            
            # Get updated preview state
            updated_preview_state = await self._get_current_preview_state()
            self.test_state['preview_visible_after'] = updated_preview_state.get('preview_visible', False)
            self.test_state['quick_view_popup_visible'] = updated_preview_state.get('quick_view_visible', False)
            
            # Check if product preview appeared
            if self.test_state['preview_visible_after']:
                self.validator.product_preview_appeared = True
                print(f"Product preview overlay appears on hover")
                
                # Check for quick view specifically
                if self.test_state['quick_view_popup_visible']:
                    self.validator.quick_view_shown = True
            
            # Evaluate the action
            action_type = action.get('action_type')
            
            if action_type == 14:  # MOVE_MOUSE action
                evaluation['valid'] = True
                evaluation['feedback'] = 'Mouse hover executed on product card'
                self.test_state['mouse_move_executed'] = True
                
                if self.test_state['preview_visible_after']:
                    evaluation['feedback'] += ' - product preview successfully appeared'
                    if self.test_state['quick_view_popup_visible']:
                        evaluation['feedback'] += ' with quick view popup'
                else:
                    evaluation['feedback'] += ' - product preview did not appear'
                
                evaluation['test_completed'] = True
                evaluation['validation_result'] = self.validator.validate_mouse_hover_result()
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected MOVE_MOUSE (14) only'
                
        except Exception as e:
            evaluation['feedback'] = f'Action execution failed: {str(e)}'
            self.logger.error(f"Failed to execute action: {e}")
        
        return evaluation
    
    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's mouse hover action in the browser backend."""
        action_type = action.get('action_type')
        
        if action_type == 14:  # MOVE_MOUSE action
            # Determine hover coordinates
            if 'coordinates' in action:
                x, y = action['coordinates']['x'], action['coordinates']['y']
            else:
                coords = self.amazon_elements_config.get('product_card', {}).get('coordinates', {'x': 0.3, 'y': 0.4})
                x, y = coords['x'], coords['y']
            
            # Execute mouse move via browser backend
            payload = {
                "x": x, 
                "y": y,
                "action": "move"
            }
            
            try:
                async with self.http_session.post(f"{self.backend_url}/move_mouse", json=payload) as response:
                    result = await response.json()
                    result['hover_coordinates'] = {'x': x, 'y': y}
                    result['action_type'] = 'move_mouse'
                    self.logger.info(f"Mouse moved to ({x}, {y})")
                    return result
            except Exception as e:
                # Fallback to click endpoint without clicking
                try:
                    payload = {"x": x, "y": y, "action": "mousemove"}
                    async with self.http_session.post(f"{self.backend_url}/click", json=payload) as response:
                        result = await response.json()
                        result['hover_coordinates'] = {'x': x, 'y': y}
                        self.logger.info(f"Mouse moved to ({x}, {y}) via fallback")
                        return result
                except Exception as e2:
                    self.logger.error(f"Mouse move execution failed: {e2}")
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
            'mouse_move_executed': self.test_state['mouse_move_executed'],
            'hover_coordinates': self.test_state['hover_coordinates'],
            'product_cards_found': self.test_state['product_cards_found'],
            'preview_visible_before': self.test_state['preview_visible_before'],
            'preview_visible_after': self.test_state['preview_visible_after'],
            'quick_view_popup_visible': self.test_state['quick_view_popup_visible'],
            'test_completed': self.test_state['mouse_move_executed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'amazon_mouse_hover_preview',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'hover_coordinates': self.validator.hover_coordinates,
            'validation_summary': self.validator.validate_mouse_hover_result(),
            'element_configuration': self.amazon_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_amazon_mouse_hover_preview(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for Amazon mouse hover preview with real browser integration.
    """
    test_env = AmazonMouseHoverTestEnvironment(url, backend_url, config_path)
    
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
def test_amazon_mouse_hover_preview_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_amazon_mouse_hover_preview(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_amazon_mouse_hover_preview_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> AmazonMouseHoverTestEnvironment:
    """Factory function to create an Amazon mouse hover preview test environment."""
    return AmazonMouseHoverTestEnvironment(url, backend_url, config_path)