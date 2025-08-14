"""
Google Maps Back Navigation Test Framework
Purpose: Test agent's ability to navigate back from route error state using browser back button
"""
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import base64

# Browser integration imports
import sys
sys.path.append('/home/ubuntu/webarena')
from browser_env.actions import create_click_action, ActionTypes

# Import base configuration loader
sys.path.append('/home/ubuntu/webarena/unit_tests/CPU')
from date_picker_plugin import ConfigLoader


class GMapsBackNavValidator:
    """Validator for Google Maps back navigation operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.navigation_actions = []
        self.back_nav_executed = False
        self.scroll_count = 0
        
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
        """Get default validation rules for navigation test."""
        return {
            'max_scroll_count': 2,
            'require_back_nav': True,
            'require_results_restoration': True
        }
    
    def log_navigation_action(self, action_type: str, action_details: Dict) -> None:
        """Log navigation action with details."""
        action_log = {
            'action_id': len(self.navigation_actions) + 1,
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'action_details': action_details
        }
        
        self.navigation_actions.append(action_log)
        self.logger.info(f"Navigation action: {action_type}")
        
        # Track specific actions
        if action_type == 'BROWSER_BACK':
            self.back_nav_executed = True
            print("Back nav executed.")  # Required validator print
        elif action_type == 'SCROLL':
            self.scroll_count += 1
    
    def validate_navigation_result(self) -> Dict[str, Any]:
        """Validate the complete navigation sequence."""
        validation_result = {
            'back_nav_executed': self.back_nav_executed,
            'scroll_count_valid': False,
            'results_restored': False,
            'scroll_count': self.scroll_count,
            'max_allowed_scrolls': self.validation_rules.get('max_scroll_count', 2),
            'errors': [],
            'warnings': []
        }
        
        # Check if back navigation was executed
        if not self.back_nav_executed:
            validation_result['errors'].append("Back navigation was not executed")
        
        # Check scroll count
        max_scrolls = self.validation_rules.get('max_scroll_count', 2)
        if self.scroll_count <= max_scrolls:
            validation_result['scroll_count_valid'] = True
        else:
            validation_result['errors'].append(f"Too many scrolls: {self.scroll_count} (max: {max_scrolls})")
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['back_nav_executed'] and
            validation_result['scroll_count_valid'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    def get_navigation_log(self) -> List[Dict]:
        """Return the complete log of navigation actions."""
        return self.navigation_actions


class GMapsBackNavTestEnvironment:
    """Test environment for Google Maps back navigation with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = GMapsBackNavValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = self.config_loader.get('test_configuration.max_steps', 3)
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 30)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.gmaps_elements_config = self.config_loader.get_section('gmaps_navigation_elements')
        
        self.test_state = {
            'current_step': 0,
            'route_error_detected': False,
            'back_nav_completed': False,
            'results_restored': False,
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
        """Get initial test setup with route error state screenshot."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show route error)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect route error panel
            error_panel_detected = await self._detect_route_error_panel()
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_error_state',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Navigate back from route error state using browser back button',
                'error_panel_detected': error_panel_detected,
                'initial_screenshot': screenshot_data,
                'navigation_elements': self.gmaps_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_action': 'BROWSER_BACK',
                    'max_scrolls': self.validator.validation_rules.get('max_scroll_count', 2)
                },
                'history': [
                    'User attempted to get directions to a café',
                    'Google Maps returned "Route not found" error',
                    'Error panel is currently displayed'
                ],
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.test_state['route_error_detected'] = error_panel_detected['found']
            self.logger.info(f"GMaps back nav test initialized: {self.url}")
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
    
    async def _detect_route_error_panel(self) -> Dict:
        """Detect route error panel on the current page."""
        error_config = self.gmaps_elements_config.get('route_error_panel', {})
        detected = {
            'found': False,
            'selector_used': None,
            'element_info': None
        }
        
        # Try each selector to find the error panel
        for selector in error_config.get('selectors', []):
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
        
        return detected
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's navigation action."""
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
            
            # Evaluate the action
            action_type = action.get('action_type')
            
            if action_type == 17:  # BROWSER_BACK (or use string identifier)
                evaluation['valid'] = True
                evaluation['feedback'] = 'Browser back navigation executed correctly'
                self.test_state['back_nav_completed'] = True
                
                # Log the navigation action
                self.validator.log_navigation_action('BROWSER_BACK', action)
                
                # Check if results page is restored
                results_restored = await self._check_results_page_restored()
                self.test_state['results_restored'] = results_restored
                
                # Test is now complete
                evaluation['test_completed'] = True
                evaluation['validation_result'] = self.validator.validate_navigation_result()
                evaluation['validation_result']['results_restored'] = results_restored
                
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected BROWSER_BACK'
                
        except Exception as e:
            evaluation['feedback'] = f'Action execution failed: {str(e)}'
            self.logger.error(f"Failed to execute action: {e}")
        
        return evaluation
    
    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's action in the browser backend."""
        action_type = action.get('action_type')
        
        if action_type == 17:  # BROWSER_BACK
            try:
                async with self.http_session.post(f"{self.backend_url}/back") as response:
                    result = await response.json()
                    self.logger.info(f"Browser back executed: {result}")
                    return result
            except Exception as e:
                self.logger.error(f"Browser back execution failed: {e}")
                raise
        else:
            raise Exception(f"Unsupported action type for this test: {action_type}")
    
    async def _check_results_page_restored(self) -> bool:
        """Check if the search results page is restored after back navigation."""
        results_config = self.gmaps_elements_config.get('results_list', {})
        
        for selector in results_config.get('selectors', []):
            try:
                eval_script = f"""
                () => {{
                    const element = document.querySelector('{selector}');
                    if (element) {{
                        const rect = element.getBoundingClientRect();
                        return {{
                            found: true,
                            visible: rect.width > 0 && rect.height > 0
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
                            return True
                        
            except Exception as e:
                self.logger.warning(f"Failed to check results selector {selector}: {e}")
                continue
        
        return False
    
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
            'route_error_detected': self.test_state['route_error_detected'],
            'back_nav_completed': self.test_state['back_nav_completed'],
            'results_restored': self.test_state['results_restored'],
            'test_completed': self.test_state['back_nav_completed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'gmaps_back_navigation',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'navigation_actions': self.validator.get_navigation_log(),
            'validation_summary': self.validator.validate_navigation_result(),
            'element_configuration': self.gmaps_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_gmaps_back_nav(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for Google Maps back navigation with real browser integration.
    
    Args:
        url: URL of the Google Maps route error page to test
        run_agent_func: External function that takes initial_observation and returns list of actions
        backend_url: Browser-in-browser backend URL (optional, uses config)
        config_path: Path to configuration file (optional)
    
    Returns:
        Complete test results dictionary
    """
    test_env = GMapsBackNavTestEnvironment(url, backend_url, config_path)
    
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
        
        # Evaluate each action (should be single BROWSER_BACK)
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
def test_gmaps_back_nav_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_gmaps_back_nav(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_gmaps_back_nav_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> GMapsBackNavTestEnvironment:
    """Factory function to create a Google Maps back navigation test environment."""
    return GMapsBackNavTestEnvironment(url, backend_url, config_path)