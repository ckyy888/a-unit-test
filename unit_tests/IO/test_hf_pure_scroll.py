"""
HuggingFace Pure Scroll Test Framework
Purpose: Test agent's ability to perform precise scroll to reveal downloads chart
Tests: SCROLL action only
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
from browser_env.actions import create_scroll_action, ActionTypes

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


class HuggingFaceScrollValidator:
    """Validator for HuggingFace scroll operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.scroll_executed = False
        self.scroll_distance = 0
        self.downloads_chart_visible = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_scroll_distance_executed(self, distance: int) -> None:
        """Print scroll distance executed."""
        self.scroll_distance = distance
        self.scroll_executed = True
        print(f"Scroll distance executed: {distance}px")
        self.logger.info(f"Scroll executed with distance: {distance}px")
    
    def validate_scroll_result(self) -> Dict[str, Any]:
        """Validate the scroll action result."""
        validation_result = {
            'scroll_executed': self.scroll_executed,
            'scroll_distance_logged': self.scroll_distance > 0,
            'downloads_chart_visible': self.downloads_chart_visible,
            'expected_distance': 600,
            'actual_distance': self.scroll_distance,
            'scroll_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if scroll was executed
        if not self.scroll_executed:
            validation_result['errors'].append("Scroll action was not executed")
        
        # Check if scroll distance was appropriate
        if self.scroll_distance < 500:
            validation_result['warnings'].append(f"Scroll distance may be insufficient: {self.scroll_distance}px")
        elif self.scroll_distance > 800:
            validation_result['warnings'].append(f"Scroll distance may be excessive: {self.scroll_distance}px")
        
        # Check if downloads chart is now visible
        if not self.downloads_chart_visible:
            validation_result['errors'].append("Downloads chart is not visible in viewport after scroll")
        
        # Determine if scroll was successful
        validation_result['scroll_successful'] = (
            self.scroll_executed and 
            self.downloads_chart_visible
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['scroll_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result


class HuggingFaceScrollTestEnvironment:
    """Test environment for HuggingFace scroll with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = HuggingFaceScrollValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 1  # Single action test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 10)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.hf_elements_config = self.config_loader.get_section('hf_scroll_elements')
        
        self.test_state = {
            'current_step': 0,
            'scroll_executed': False,
            'scroll_distance': 0,
            'initial_scroll_position': 0,
            'final_scroll_position': 0,
            'downloads_chart_before_scroll': False,
            'downloads_chart_after_scroll': False,
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
        """Get initial test setup with model page loaded at top, downloads chart below fold."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show HuggingFace model page)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect downloads chart visibility and page state
            chart_detection = await self._detect_downloads_chart()
            page_state = await self._get_current_page_state()
            
            # Store initial scroll position and chart visibility
            self.test_state['initial_scroll_position'] = page_state.get('scroll_position', 0)
            self.test_state['downloads_chart_before_scroll'] = chart_detection.get('found', False)
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_page_top',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Execute single scroll down to reveal downloads chart below fold',
                'chart_detection': chart_detection,
                'page_state': page_state,
                'initial_screenshot': screenshot_data,
                'scroll_elements': self.hf_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_action': 'SCROLL_DOWN(600px) once only',
                    'expected_result': 'downloads chart now visible in viewport'
                },
                'history': [
                    'HuggingFace model page loaded',
                    'Page positioned at top',
                    'Model description visible',
                    'Downloads chart below fold (not visible)',
                    'Ready for scroll action'
                ],
                'current_viewport': {
                    'scroll_position': self.test_state['initial_scroll_position'],
                    'downloads_chart_visible': self.test_state['downloads_chart_before_scroll'],
                    'chart_expected_after_scroll': True
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"HuggingFace scroll test initialized: {self.url}")
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
    
    async def _detect_downloads_chart(self) -> Dict:
        """Detect downloads chart visibility in viewport."""
        chart_config = self.hf_elements_config.get('downloads_chart', {})
        detected = {
            'found': False,
            'selector_used': None,
            'in_viewport': False,
            'element_info': None
        }
        
        for selector in chart_config.get('selectors', []):
            try:
                eval_script = f"""
                () => {{
                    const element = document.querySelector('{selector}');
                    if (element) {{
                        const rect = element.getBoundingClientRect();
                        const viewportHeight = window.innerHeight;
                        const viewportWidth = window.innerWidth;
                        
                        return {{
                            found: true,
                            selector: '{selector}',
                            coordinates: {{
                                x: rect.left + rect.width / 2,
                                y: rect.top + rect.height / 2
                            }},
                            visible: rect.width > 0 && rect.height > 0,
                            in_viewport: rect.top >= 0 && rect.top <= viewportHeight && 
                                        rect.left >= 0 && rect.left <= viewportWidth,
                            rect: {{
                                top: rect.top,
                                bottom: rect.bottom,
                                left: rect.left,
                                right: rect.right
                            }},
                            viewport_height: viewportHeight,
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
                                'in_viewport': result.get('in_viewport', False),
                                'element_info': result
                            }
                            break
                    
            except Exception as e:
                self.logger.warning(f"Failed to check selector {selector}: {e}")
                continue
        
        return detected
    
    async def _get_current_page_state(self) -> Dict:
        """Get current page scroll position and viewport info."""
        try:
            eval_script = """
            () => {
                return {
                    scroll_position: window.pageYOffset,
                    scroll_height: document.body.scrollHeight,
                    viewport_height: window.innerHeight,
                    viewport_width: window.innerWidth,
                    can_scroll_down: window.pageYOffset < (document.body.scrollHeight - window.innerHeight)
                };
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
            self.logger.error(f"Failed to get page state: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's scroll action."""
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
            
            # Log scroll distance
            scroll_distance = self._extract_scroll_distance(action, browser_result)
            if scroll_distance:
                self.validator.log_scroll_distance_executed(scroll_distance)
                self.test_state['scroll_distance'] = scroll_distance
            
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
            
            # Get updated page state and chart visibility
            await asyncio.sleep(0.5)  # Wait for scroll to complete
            updated_page_state = await self._get_current_page_state()
            updated_chart_detection = await self._detect_downloads_chart()
            
            self.test_state['final_scroll_position'] = updated_page_state.get('scroll_position', 0)
            self.test_state['downloads_chart_after_scroll'] = updated_chart_detection.get('in_viewport', False)
            
            # Evaluate the action
            action_type = action.get('action_type')
            
            if action_type == 7:  # SCROLL action
                evaluation['valid'] = True
                evaluation['feedback'] = f'Scroll executed - distance: {scroll_distance}px'
                self.test_state['scroll_executed'] = True
                
                # Check if downloads chart is now visible
                if self.test_state['downloads_chart_after_scroll']:
                    self.validator.downloads_chart_visible = True
                    print(f"Downloads chart now visible in viewport")
                    evaluation['feedback'] += ' - downloads chart successfully revealed'
                else:
                    evaluation['feedback'] += ' - downloads chart still not visible'
                
                evaluation['test_completed'] = True
                evaluation['validation_result'] = self.validator.validate_scroll_result()
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected SCROLL (7) only'
                
        except Exception as e:
            evaluation['feedback'] = f'Action execution failed: {str(e)}'
            self.logger.error(f"Failed to execute action: {e}")
        
        return evaluation
    
    def _extract_scroll_distance(self, action: Dict, browser_result: Dict) -> int:
        """Extract scroll distance from action or browser result."""
        # Try to get distance from action
        if 'coordinate' in action and isinstance(action['coordinate'], list) and len(action['coordinate']) >= 2:
            return abs(action['coordinate'][1])  # y-coordinate for vertical scroll
        
        # Try to get from browser result
        if browser_result and 'scroll_distance' in browser_result:
            return browser_result['scroll_distance']
        
        # Calculate from scroll position change
        position_change = self.test_state['final_scroll_position'] - self.test_state['initial_scroll_position']
        if position_change > 0:
            return int(position_change)
        
        # Default expected distance
        return 600
    
    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's action in the browser backend."""
        action_type = action.get('action_type')
        
        if action_type == 7:  # SCROLL action
            # Determine scroll distance and direction
            scroll_distance = 600  # Default scroll distance
            
            if 'coordinate' in action and isinstance(action['coordinate'], list) and len(action['coordinate']) >= 2:
                scroll_distance = abs(action['coordinate'][1])  # y-coordinate
            elif 'scroll_distance' in action:
                scroll_distance = action['scroll_distance']
            
            # Execute scroll via browser backend
            payload = {"dx": 0, "dy": scroll_distance}  # Scroll down
            
            try:
                async with self.http_session.post(f"{self.backend_url}/scroll", json=payload) as response:
                    result = await response.json()
                    result['scroll_distance'] = scroll_distance
                    self.logger.info(f"Scroll executed: {scroll_distance}px down")
                    return result
            except Exception as e:
                self.logger.error(f"Scroll execution failed: {e}")
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
            'scroll_executed': self.test_state['scroll_executed'],
            'scroll_distance': self.test_state['scroll_distance'],
            'initial_scroll_position': self.test_state['initial_scroll_position'],
            'final_scroll_position': self.test_state['final_scroll_position'],
            'downloads_chart_before_scroll': self.test_state['downloads_chart_before_scroll'],
            'downloads_chart_after_scroll': self.test_state['downloads_chart_after_scroll'],
            'test_completed': self.test_state['scroll_executed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'hf_pure_scroll',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'scroll_distance': self.validator.scroll_distance,
            'validation_summary': self.validator.validate_scroll_result(),
            'element_configuration': self.hf_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_hf_pure_scroll(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for HuggingFace pure scroll with real browser integration.
    """
    test_env = HuggingFaceScrollTestEnvironment(url, backend_url, config_path)
    
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
def test_hf_pure_scroll_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_hf_pure_scroll(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_hf_pure_scroll_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> HuggingFaceScrollTestEnvironment:
    """Factory function to create a HuggingFace pure scroll test environment."""
    return HuggingFaceScrollTestEnvironment(url, backend_url, config_path)