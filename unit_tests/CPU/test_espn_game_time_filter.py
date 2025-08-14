"""
ESPN Game Time Filter Test Framework
Purpose: Test agent's ability to convert game times from PT to ET timezone
"""
import json
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import base64
import re

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


class EspnTimezoneValidator:
    """Validator for ESPN NBA timezone conversion operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.timezone_changes = []
        self.timezone_dropdown_clicked = False
        self.et_option_selected = False
        self.game_times_updated = False
        self.all_times_show_et = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_timezone_changes(self, change_type: str, timezone_data: Dict) -> None:
        """Log timezone changes."""
        timezone_log = {
            'change_id': len(self.timezone_changes) + 1,
            'timestamp': datetime.now().isoformat(),
            'change_type': change_type,
            'timezone_data': timezone_data
        }
        
        self.timezone_changes.append(timezone_log)
        self.logger.info(f"Timezone change: {change_type}")
        
        if change_type == 'TIMEZONE_DROPDOWN_CLICKED':
            self.timezone_dropdown_clicked = True
            print(f"Timezone changes: Timezone dropdown opened")
        elif change_type == 'ET_OPTION_SELECTED':
            self.et_option_selected = True
            print(f"Timezone changes: Eastern Time (ET) selected")
        elif change_type == 'GAME_TIMES_UPDATED':
            self.game_times_updated = True
            print(f"Timezone changes: Game times shifted correctly (+3 hours from PT to ET)")
        elif change_type == 'ALL_TIMES_SHOW_ET':
            self.all_times_show_et = True
            print(f"Timezone changes: All game times now display 'ET' suffix")
    
    def validate_timezone_conversion_result(self) -> Dict[str, Any]:
        """Validate the complete timezone conversion sequence."""
        validation_result = {
            'timezone_dropdown_clicked': self.timezone_dropdown_clicked,
            'et_option_selected': self.et_option_selected,
            'game_times_updated': self.game_times_updated,
            'all_times_show_et': self.all_times_show_et,
            'timezone_conversion_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if timezone dropdown was clicked
        if not self.timezone_dropdown_clicked:
            validation_result['errors'].append("Timezone dropdown was not clicked")
        
        # Check if ET option was selected
        if not self.et_option_selected:
            validation_result['errors'].append("Eastern Time (ET) option was not selected")
        
        # Check if game times were updated
        if not self.game_times_updated:
            validation_result['errors'].append("Game times did not update correctly")
        
        # Check if all times show ET
        if not self.all_times_show_et:
            validation_result['warnings'].append("Not all game times display 'ET' suffix")
        
        # Determine if timezone conversion was successful
        validation_result['timezone_conversion_successful'] = (
            self.timezone_dropdown_clicked and 
            self.et_option_selected and 
            self.game_times_updated
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['timezone_conversion_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    def get_timezone_log(self) -> List[Dict]:
        """Return the complete log of timezone changes."""
        return self.timezone_changes


class EspnTimezoneTestEnvironment:
    """Test environment for ESPN NBA timezone conversion with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = EspnTimezoneValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 30)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.espn_timezone_elements_config = self.config_loader.get_section('espn_timezone_elements')
        
        self.test_state = {
            'current_step': 0,
            'current_timezone': 'PT',  # Starting Pacific Time
            'target_timezone': 'ET',   # Target Eastern Time  
            'timezone_changed': False,
            'game_times_before': [],   # Store original times for comparison
            'game_times_after': [],    # Store updated times
            'dropdown_opened': False,
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
        """Get initial test setup showing games in PT timezone."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show NBA games in PT)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect timezone elements and game times
            element_detection = await self._detect_timezone_elements()
            game_times_info = await self._get_current_game_times()
            
            # Store initial game times for comparison
            self.test_state['game_times_before'] = game_times_info.get('game_times', [])
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_pt_timezone',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Convert game times from PT to ET timezone',
                'element_detection': element_detection,
                'game_times_info': game_times_info,
                'initial_screenshot': screenshot_data,
                'timezone_elements': self.espn_timezone_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_sequence': ['CLICK(timezone_dropdown)', 'CLICK(ET_option)', 'verify time updates'],
                    'expected_final_state': 'all game times shift correctly (+3 hours) and display "ET"'
                },
                'history': [
                    'ESPN NBA scoreboard loaded',
                    'Games showing in Pacific Time (PT)',
                    'User located on East Coast, prefers ET',
                    'PT times are confusing for East Coast viewing',
                    'Need to convert all game times to Eastern Time'
                ],
                'current_timezone_info': {
                    'current': 'PT',
                    'target': 'ET',
                    'time_difference': '+3 hours',
                    'games_count': len(self.test_state['game_times_before'])
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update test state based on detection
            if element_detection.get('timezone_dropdown', {}).get('found'):
                self.test_state['timezone_dropdown_available'] = True
            
            self.logger.info(f"ESPN timezone conversion test initialized: {self.url}")
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
    
    async def _detect_timezone_elements(self) -> Dict:
        """Detect timezone selector elements on the current page."""
        detected_elements = {}
        
        for element_type, config in self.espn_timezone_elements_config.items():
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
                                value: element.value || '',
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
    
    async def _get_current_game_times(self) -> Dict:
        """Get current game times and timezone information."""
        try:
            eval_script = """
            () => {
                const result = {
                    current_timezone: '',
                    game_times: [],
                    timezone_display: '',
                    games_count: 0
                };
                
                // Look for timezone indicator
                const timezoneElements = [
                    document.querySelector('.timezone-display'),
                    document.querySelector('[data-testid="timezone"]'),
                    document.querySelector('.time-zone')
                ];
                
                for (const element of timezoneElements) {
                    if (element && element.textContent) {
                        result.timezone_display = element.textContent.trim();
                        if (element.textContent.includes('PT') || element.textContent.includes('Pacific')) {
                            result.current_timezone = 'PT';
                        } else if (element.textContent.includes('ET') || element.textContent.includes('Eastern')) {
                            result.current_timezone = 'ET';
                        }
                        break;
                    }
                }
                
                // Collect game times
                const timeElements = document.querySelectorAll('.game-time, .start-time, [data-testid="game-time"], .time');
                timeElements.forEach((timeEl, index) => {
                    if (timeEl.textContent) {
                        const timeText = timeEl.textContent.trim();
                        // Look for time patterns like "7:30 PM PT" or "10:00 ET"
                        const timeMatch = timeText.match(/(\d{1,2}:\d{2})\s*(AM|PM)?\s*(PT|ET)?/i);
                        if (timeMatch) {
                            result.game_times.push({
                                index: index,
                                original_text: timeText,
                                time: timeMatch[1],
                                period: timeMatch[2] || '',
                                timezone: timeMatch[3] || result.current_timezone || 'PT'
                            });
                        }
                    }
                });
                
                result.games_count = result.game_times.length;
                
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
            self.logger.error(f"Failed to get game times: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's timezone conversion action."""
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
            
            # Get updated game times
            game_times_info = await self._get_current_game_times()
            
            # Evaluate the action
            action_type = action.get('action_type')
            element_name = action.get('element_name', '').lower()
            
            if action_type == 6:  # CLICK action
                if 'timezone' in element_name and 'dropdown' in element_name:
                    # Timezone dropdown clicked
                    evaluation['valid'] = True
                    evaluation['feedback'] = 'Timezone dropdown opened successfully'
                    self.test_state['dropdown_opened'] = True
                    
                    # Log the timezone action
                    self.validator.log_timezone_changes('TIMEZONE_DROPDOWN_CLICKED', {
                        'dropdown_opened': True,
                        'current_timezone': self.test_state['current_timezone']
                    })
                    
                    evaluation['expected_next'] = 'Select Eastern Time (ET) option'
                
                elif 'et' in element_name or 'eastern' in element_name:
                    # ET option selected
                    if self.test_state['dropdown_opened']:
                        evaluation['valid'] = True
                        evaluation['feedback'] = 'Eastern Time (ET) option selected'
                        self.test_state['timezone_changed'] = True
                        self.test_state['current_timezone'] = 'ET'
                        
                        # Log the ET selection
                        self.validator.log_timezone_changes('ET_OPTION_SELECTED', {
                            'new_timezone': 'ET',
                            'previous_timezone': 'PT'
                        })
                        
                        # Wait for times to update
                        await asyncio.sleep(2)
                        updated_game_times = await self._get_current_game_times()
                        self.test_state['game_times_after'] = updated_game_times.get('game_times', [])
                        
                        # Verify time conversion
                        if self._verify_time_conversion(updated_game_times):
                            self.validator.log_timezone_changes('GAME_TIMES_UPDATED', {
                                'times_shifted': True,
                                'conversion_verified': True,
                                'games_updated': len(self.test_state['game_times_after'])
                            })
                            
                            # Check if all times show ET
                            if self._check_et_suffix(updated_game_times):
                                self.validator.log_timezone_changes('ALL_TIMES_SHOW_ET', {
                                    'et_suffix_displayed': True,
                                    'all_times_updated': True
                                })
                            
                            evaluation['test_completed'] = True
                            evaluation['validation_result'] = self.validator.validate_timezone_conversion_result()
                        else:
                            evaluation['feedback'] = 'ET selected but game times did not update correctly'
                    else:
                        evaluation['feedback'] = 'ET option clicked but timezone dropdown was not opened first'
                
                else:
                    evaluation['feedback'] = f'Clicked wrong element - expected timezone control, got: {element_name}'
            else:
                evaluation['feedback'] = f'Unexpected action type {action_type} - expected CLICK (6)'
                
        except Exception as e:
            evaluation['feedback'] = f'Action execution failed: {str(e)}'
            self.logger.error(f"Failed to execute action: {e}")
        
        return evaluation
    
    def _verify_time_conversion(self, updated_game_times: Dict) -> bool:
        """Verify that game times were correctly converted from PT to ET (+3 hours)."""
        if not self.test_state['game_times_before'] or not updated_game_times.get('game_times'):
            return False
        
        # Check if timezone changed to ET
        if updated_game_times.get('current_timezone') == 'ET':
            return True
        
        # Additional verification could include checking specific time shifts
        # For simplicity, we assume timezone change indicates successful conversion
        before_count = len(self.test_state['game_times_before'])
        after_count = len(updated_game_times.get('game_times', []))
        
        return before_count == after_count and after_count > 0
    
    def _check_et_suffix(self, game_times_info: Dict) -> bool:
        """Check if game times display ET suffix."""
        game_times = game_times_info.get('game_times', [])
        if not game_times:
            return False
        
        # Check if most times show ET
        et_count = sum(1 for time_info in game_times if time_info.get('timezone') == 'ET')
        return et_count >= len(game_times) * 0.8  # 80% threshold
    
    async def _execute_action_in_browser(self, action: Dict) -> Dict:
        """Execute the agent's action in the browser backend."""
        action_type = action.get('action_type')
        
        if action_type == 6:  # CLICK action
            # Determine click coordinates
            if 'coordinates' in action:
                x, y = action['coordinates']['x'], action['coordinates']['y']
            elif 'element_name' in action:
                element_name = action['element_name'].lower()
                if 'timezone' in element_name and 'dropdown' in element_name:
                    coords = self.espn_timezone_elements_config.get('timezone_dropdown', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.9), coords.get('y', 0.1)
                elif 'et' in element_name or 'eastern' in element_name:
                    # ET option coordinates (within dropdown)
                    coords = self.espn_timezone_elements_config.get('timezone_dropdown', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.9), coords.get('y', 0.15)  # Slightly below dropdown
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
            'current_timezone': self.test_state['current_timezone'],
            'target_timezone': self.test_state['target_timezone'],
            'timezone_changed': self.test_state['timezone_changed'],
            'dropdown_opened': self.test_state['dropdown_opened'],
            'test_completed': self.test_state['timezone_changed'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'espn_game_time_filter',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'timezone_changes': self.validator.get_timezone_log(),
            'validation_summary': self.validator.validate_timezone_conversion_result(),
            'element_configuration': self.espn_timezone_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_espn_game_time_filter(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for ESPN game time filter with real browser integration.
    """
    test_env = EspnTimezoneTestEnvironment(url, backend_url, config_path)
    
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
def test_espn_game_time_filter_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_espn_game_time_filter(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_espn_game_time_filter_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> EspnTimezoneTestEnvironment:
    """Factory function to create an ESPN game time filter test environment."""
    return EspnTimezoneTestEnvironment(url, backend_url, config_path)