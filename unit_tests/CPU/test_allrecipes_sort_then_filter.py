"""
AllRecipes Sort Then Filter Test Framework
Purpose: Test agent's ability to sort recipes by rating and then apply time filter
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


class AllRecipesValidator:
    """Validator for AllRecipes sort and filter operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.sort_filter_applications = []
        self.sort_dropdown_clicked = False
        self.rating_option_selected = False
        self.time_filter_applied = False
        self.url_parameters_correct = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_sort_and_filter_application(self, action_type: str, filter_data: Dict) -> None:
        """Log sort and filter applications."""
        filter_log = {
            'action_id': len(self.sort_filter_applications) + 1,
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'filter_data': filter_data
        }
        
        self.sort_filter_applications.append(filter_log)
        self.logger.info(f"Sort/Filter action: {action_type}")
        
        if action_type == 'SORT_DROPDOWN_CLICKED':
            self.sort_dropdown_clicked = True
            print(f"Sort and filter applications: Sort dropdown opened")
        elif action_type == 'RATING_OPTION_SELECTED':
            self.rating_option_selected = True
            print(f"Sort and filter applications: Sort by rating selected")
        elif action_type == 'TIME_FILTER_APPLIED':
            self.time_filter_applied = True
            print(f"Sort and filter applications: Under 30 min filter applied")
        elif action_type == 'URL_PARAMETERS_VERIFIED':
            self.url_parameters_correct = True
            print(f"Sort and filter applications: URL parameters verified - {filter_data.get('url_params', 'sort=rating&time=30min')}")
    
    def validate_sort_filter_result(self) -> Dict[str, Any]:
        """Validate the complete sort and filter sequence."""
        validation_result = {
            'sort_dropdown_clicked': self.sort_dropdown_clicked,
            'rating_option_selected': self.rating_option_selected,
            'time_filter_applied': self.time_filter_applied,
            'url_parameters_correct': self.url_parameters_correct,
            'sort_filter_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if sort dropdown was clicked
        if not self.sort_dropdown_clicked:
            validation_result['errors'].append("Sort dropdown was not clicked")
        
        # Check if rating option was selected
        if not self.rating_option_selected:
            validation_result['errors'].append("Sort by rating option was not selected")
        
        # Check if time filter was applied
        if not self.time_filter_applied:
            validation_result['errors'].append("Under 30 min time filter was not applied")
        
        # Check if URL parameters are correct
        if not self.url_parameters_correct:
            validation_result['warnings'].append("URL does not contain both sort=rating&time=30min parameters")
        
        # Determine if sort and filter was successful
        validation_result['sort_filter_successful'] = (
            self.sort_dropdown_clicked and 
            self.rating_option_selected and 
            self.time_filter_applied
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['sort_filter_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    def get_sort_filter_log(self) -> List[Dict]:
        """Return the complete log of sort and filter applications."""
        return self.sort_filter_applications


class AllRecipesTestEnvironment:
    """Test environment for AllRecipes sort then filter with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = AllRecipesValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 30)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.allrecipes_elements_config = self.config_loader.get_section('allrecipes_elements')
        
        self.test_state = {
            'current_step': 0,
            'current_sort': 'relevance',  # Starting sort order
            'sort_dropdown_open': False,
            'rating_sort_applied': False,
            'time_filter_applied': False,
            'results_updated': False,
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
        """Get initial test setup with unsorted chicken recipe results."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show unsorted chicken recipes)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect sort and filter elements
            element_detection = await self._detect_sort_filter_elements()
            page_state = await self._get_current_page_state()
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_unsorted_results',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Sort recipes by rating, then apply under 30 min time filter',
                'element_detection': element_detection,
                'page_state': page_state,
                'initial_screenshot': screenshot_data,
                'sort_filter_elements': self.allrecipes_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_sequence': ['CLICK(sort_dropdown)', 'CLICK(rating_option)', 'wait', 'CLICK(time_filter)'],
                    'expected_final_state': 'URL parameters include both sort=rating&time=30min'
                },
                'history': [
                    'AllRecipes search page loaded',
                    'Searched for "chicken recipes"',
                    'Results displayed in default order (unsorted)',
                    'Recipe grid and filter sidebar are visible',
                    'Sort dropdown shows current selection',
                    'Time filter options available in sidebar'
                ],
                'current_results': {
                    'sort_order': 'relevance',
                    'time_filter': None,
                    'recipe_count': page_state.get('recipe_count', 'multiple')
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update test state based on detection
            if element_detection.get('sort_dropdown', {}).get('found'):
                self.test_state['sort_dropdown_available'] = True
            if element_detection.get('time_filter', {}).get('found'):
                self.test_state['time_filter_available'] = True
            
            self.logger.info(f"AllRecipes sort then filter test initialized: {self.url}")
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
    
    async def _detect_sort_filter_elements(self) -> Dict:
        """Detect sort and filter elements on the current page."""
        detected_elements = {}
        
        for element_type, config in self.allrecipes_elements_config.items():
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
                                selected: element.selected || false,
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
    
    async def _get_current_page_state(self) -> Dict:
        """Get current page state including sort order and filters."""
        try:
            eval_script = """
            () => {
                const result = {
                    current_url: window.location.href,
                    sort_parameter: null,
                    time_parameter: null,
                    recipe_count: 0,
                    current_sort_text: '',
                    filters_applied: []
                };
                
                // Check URL parameters
                const urlParams = new URLSearchParams(window.location.search);
                result.sort_parameter = urlParams.get('sort');
                result.time_parameter = urlParams.get('time') || urlParams.get('duration');
                
                // Get current sort dropdown text
                const sortDropdown = document.querySelector('.sort-dropdown, [data-testid="sort-select"], select[name="sort"]');
                if (sortDropdown) {
                    result.current_sort_text = sortDropdown.textContent || sortDropdown.value || '';
                }
                
                // Count recipe cards
                const recipeCards = document.querySelectorAll('.recipe-card, .card-recipe, [data-testid="recipe-card"]');
                result.recipe_count = recipeCards.length;
                
                // Check applied filters
                const activeFilters = document.querySelectorAll('.filter-active, .applied-filter, input[type="checkbox"]:checked');
                activeFilters.forEach(filter => {
                    if (filter.value || filter.textContent) {
                        result.filters_applied.push(filter.value || filter.textContent.trim());
                    }
                });
                
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
            self.logger.error(f"Failed to get page state: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's sort and filter action."""
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
            
            # Get updated page state
            page_state = await self._get_current_page_state()
            
            # Evaluate the action based on current step
            action_type = action.get('action_type')
            element_name = action.get('element_name', '').lower()
            
            if action_type == 6:  # CLICK action
                if 'sort' in element_name and 'dropdown' in element_name:
                    # Sort dropdown clicked
                    evaluation['valid'] = True
                    evaluation['feedback'] = 'Sort dropdown opened successfully'
                    self.test_state['sort_dropdown_open'] = True
                    
                    # Log the sort action
                    self.validator.log_sort_and_filter_application('SORT_DROPDOWN_CLICKED', {
                        'dropdown_opened': True
                    })
                    
                    evaluation['expected_next'] = 'Select rating option from dropdown'
                
                elif 'rating' in element_name:
                    # Rating option selected
                    if self.test_state['sort_dropdown_open']:
                        evaluation['valid'] = True
                        evaluation['feedback'] = 'Sort by rating option selected'
                        self.test_state['rating_sort_applied'] = True
                        self.test_state['current_sort'] = 'rating'
                        
                        # Log the rating selection
                        self.validator.log_sort_and_filter_application('RATING_OPTION_SELECTED', {
                            'sort_by': 'rating',
                            'sort_applied': True
                        })
                        
                        evaluation['expected_next'] = 'Wait for results to update, then apply time filter'
                    else:
                        evaluation['feedback'] = 'Rating option clicked but sort dropdown was not opened first'
                
                elif 'time' in element_name or '30' in element_name or 'filter' in element_name:
                    # Time filter applied
                    if self.test_state['rating_sort_applied']:
                        evaluation['valid'] = True
                        evaluation['feedback'] = 'Under 30 min time filter applied successfully'
                        self.test_state['time_filter_applied'] = True
                        
                        # Log the filter application
                        self.validator.log_sort_and_filter_application('TIME_FILTER_APPLIED', {
                            'time_filter': '30min',
                            'filter_applied': True
                        })
                        
                        # Check URL parameters
                        await asyncio.sleep(2)  # Wait for URL update
                        updated_page_state = await self._get_current_page_state()
                        
                        if self._check_url_parameters(updated_page_state):
                            self.validator.log_sort_and_filter_application('URL_PARAMETERS_VERIFIED', {
                                'url_params': f"sort={updated_page_state.get('sort_parameter', 'rating')}&time={updated_page_state.get('time_parameter', '30min')}",
                                'url': updated_page_state.get('current_url', '')
                            })
                        
                        evaluation['test_completed'] = True
                        evaluation['validation_result'] = self.validator.validate_sort_filter_result()
                    else:
                        evaluation['feedback'] = 'Time filter clicked but rating sort was not applied first'
                
                else:
                    evaluation['feedback'] = f'Clicked wrong element - expected sort/filter control, got: {element_name}'
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
                if 'sort' in element_name and 'dropdown' in element_name:
                    coords = self.allrecipes_elements_config.get('sort_dropdown', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.8), coords.get('y', 0.2)
                elif 'rating' in element_name:
                    # Rating option coordinates (within dropdown)
                    coords = self.allrecipes_elements_config.get('sort_dropdown', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.8), coords.get('y', 0.3)  # Slightly below dropdown
                elif 'time' in element_name or '30' in element_name:
                    coords = self.allrecipes_elements_config.get('time_filter', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.1), coords.get('y', 0.4)
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
    
    def _check_url_parameters(self, page_state: Dict) -> bool:
        """Check if URL contains both sort=rating and time=30min parameters."""
        sort_param = page_state.get('sort_parameter')
        time_param = page_state.get('time_parameter')
        
        # Check for both parameters
        has_rating_sort = sort_param and ('rating' in str(sort_param).lower())
        has_time_filter = time_param and ('30' in str(time_param) or 'min' in str(time_param).lower())
        
        return has_rating_sort and has_time_filter
    
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
            'current_sort': self.test_state['current_sort'],
            'sort_dropdown_open': self.test_state['sort_dropdown_open'],
            'rating_sort_applied': self.test_state['rating_sort_applied'],
            'time_filter_applied': self.test_state['time_filter_applied'],
            'test_completed': self.test_state['time_filter_applied'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'allrecipes_sort_then_filter',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'sort_filter_applications': self.validator.get_sort_filter_log(),
            'validation_summary': self.validator.validate_sort_filter_result(),
            'element_configuration': self.allrecipes_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_allrecipes_sort_then_filter(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for AllRecipes sort then filter with real browser integration.
    """
    test_env = AllRecipesTestEnvironment(url, backend_url, config_path)
    
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
def test_allrecipes_sort_then_filter_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_allrecipes_sort_then_filter(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_allrecipes_sort_then_filter_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> AllRecipesTestEnvironment:
    """Factory function to create an AllRecipes sort then filter test environment."""
    return AllRecipesTestEnvironment(url, backend_url, config_path)