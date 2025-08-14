"""
HuggingFace Model Comparison Test Framework
Purpose: Test agent's ability to compare BERT Base vs BERT Large models
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


class HuggingFaceValidator:
    """Validator for HuggingFace model comparison operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.model_selections = []
        self.bert_base_selected = False
        self.bert_large_selected = False
        self.compare_button_clicked = False
        self.comparison_table_visible = False
        self.metrics_displayed = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_selected_models(self, selection_type: str, model_data: Dict) -> None:
        """Log model selections."""
        selection_log = {
            'selection_id': len(self.model_selections) + 1,
            'timestamp': datetime.now().isoformat(),
            'selection_type': selection_type,
            'model_data': model_data
        }
        
        self.model_selections.append(selection_log)
        self.logger.info(f"Model selection: {selection_type}")
        
        if selection_type == 'BERT_BASE_SELECTED':
            self.bert_base_selected = True
            print(f"Selected models: BERT Base model selected for comparison")
        elif selection_type == 'BERT_LARGE_SELECTED':
            self.bert_large_selected = True
            print(f"Selected models: BERT Large model selected for comparison")
        elif selection_type == 'COMPARE_BUTTON_CLICKED':
            self.compare_button_clicked = True
            print(f"Selected models: Compare button clicked - generating comparison table")
        elif selection_type == 'COMPARISON_TABLE_DISPLAYED':
            self.comparison_table_visible = True
            self.metrics_displayed = True
            print(f"Selected models: Comparison table shows parameter count and performance metrics")
    
    def validate_model_comparison_result(self) -> Dict[str, Any]:
        """Validate the complete model comparison sequence."""
        validation_result = {
            'bert_base_selected': self.bert_base_selected,
            'bert_large_selected': self.bert_large_selected,
            'compare_button_clicked': self.compare_button_clicked,
            'comparison_table_visible': self.comparison_table_visible,
            'metrics_displayed': self.metrics_displayed,
            'model_comparison_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if BERT Base was selected
        if not self.bert_base_selected:
            validation_result['errors'].append("BERT Base model was not selected")
        
        # Check if BERT Large was selected
        if not self.bert_large_selected:
            validation_result['errors'].append("BERT Large model was not selected")
        
        # Check if compare button was clicked
        if not self.compare_button_clicked:
            validation_result['errors'].append("Compare button was not clicked")
        
        # Check if comparison table is visible
        if not self.comparison_table_visible:
            validation_result['errors'].append("Comparison table is not visible")
        
        # Check if metrics are displayed
        if not self.metrics_displayed:
            validation_result['errors'].append("Model comparison metrics are not displayed")
        
        # Determine if model comparison was successful
        validation_result['model_comparison_successful'] = (
            self.bert_base_selected and 
            self.bert_large_selected and 
            self.compare_button_clicked and
            self.comparison_table_visible
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['model_comparison_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    def get_selection_log(self) -> List[Dict]:
        """Return the complete log of model selections."""
        return self.model_selections


class HuggingFaceTestEnvironment:
    """Test environment for HuggingFace model comparison with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = HuggingFaceValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 30)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.huggingface_elements_config = self.config_loader.get_section('huggingface_elements')
        
        self.test_state = {
            'current_step': 0,
            'bert_base_selected': False,
            'bert_large_selected': False, 
            'compare_button_clicked': False,
            'comparison_table_visible': False,
            'metrics_displayed': False,
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
        """Get initial test setup showing BERT model variants."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show BERT model search results)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect model comparison elements
            element_detection = await self._detect_model_elements()
            models_info = await self._get_current_models_info()
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_bert_models',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Compare BERT Base vs BERT Large models for size and performance',
                'element_detection': element_detection,
                'models_info': models_info,
                'initial_screenshot': screenshot_data,
                'model_elements': self.huggingface_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_sequence': ['CLICK(bert_base_checkbox)', 'CLICK(bert_large_checkbox)', 'CLICK(compare_button)'],
                    'expected_final_state': 'comparison table shows parameter count and performance metrics'
                },
                'history': [
                    'HuggingFace model search page loaded',
                    'Multiple BERT variants listed',
                    'Need efficiency comparison between Base and Large',
                    'Model cards show basic information',
                    'Comparison checkboxes available for selection'
                ],
                'available_models': {
                    'bert_base': 'Available for comparison',
                    'bert_large': 'Available for comparison',
                    'comparison_enabled': True
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update test state based on detection
            if element_detection.get('bert_base_checkbox', {}).get('found'):
                self.test_state['bert_base_available'] = True
            if element_detection.get('bert_large_checkbox', {}).get('found'):
                self.test_state['bert_large_available'] = True
            
            self.logger.info(f"HuggingFace model comparison test initialized: {self.url}")
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
    
    async def _detect_model_elements(self) -> Dict:
        """Detect model comparison elements on the current page."""
        detected_elements = {}
        
        for element_type, config in self.huggingface_elements_config.items():
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
                                checked: element.checked || false,
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
    
    async def _get_current_models_info(self) -> Dict:
        """Get current model selection state and comparison status."""
        try:
            eval_script = """
            () => {
                const result = {
                    bert_base_checked: false,
                    bert_large_checked: false,
                    compare_button_enabled: false,
                    comparison_table_visible: false,
                    models_available: 0
                };
                
                // Check BERT Base checkbox
                const bertBaseCheckbox = document.querySelector('[data-model="bert-base"] input[type="checkbox"]') ||
                                       document.querySelector('.model-card[data-name*="bert-base"] .compare-checkbox');
                if (bertBaseCheckbox) {
                    result.bert_base_checked = bertBaseCheckbox.checked;
                }
                
                // Check BERT Large checkbox
                const bertLargeCheckbox = document.querySelector('[data-model="bert-large"] input[type="checkbox"]') ||
                                        document.querySelector('.model-card[data-name*="bert-large"] .compare-checkbox');
                if (bertLargeCheckbox) {
                    result.bert_large_checked = bertLargeCheckbox.checked;
                }
                
                // Check compare button
                const compareButton = document.querySelector('.compare-models-btn') ||
                                    document.querySelector('[data-testid="compare-button"]');
                if (compareButton) {
                    result.compare_button_enabled = !compareButton.disabled;
                }
                
                // Check comparison table
                const comparisonTable = document.querySelector('.comparison-table') ||
                                      document.querySelector('.model-comparison') ||
                                      document.querySelector('[data-testid="comparison-table"]');
                if (comparisonTable) {
                    const rect = comparisonTable.getBoundingClientRect();
                    result.comparison_table_visible = rect.width > 0 && rect.height > 0;
                }
                
                // Count available models
                const modelCards = document.querySelectorAll('.model-card, [data-testid="model-card"]');
                result.models_available = modelCards.length;
                
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
            self.logger.error(f"Failed to get models info: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's model comparison action."""
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
            
            # Get updated models state
            models_info = await self._get_current_models_info()
            
            # Evaluate the action
            action_type = action.get('action_type')
            element_name = action.get('element_name', '').lower()
            
            if action_type == 6:  # CLICK action
                if 'bert' in element_name and 'base' in element_name and 'checkbox' in element_name:
                    # BERT Base checkbox clicked
                    evaluation['valid'] = True
                    evaluation['feedback'] = 'BERT Base model selected for comparison'
                    self.test_state['bert_base_selected'] = True
                    
                    # Log the model selection
                    self.validator.log_selected_models('BERT_BASE_SELECTED', {
                        'model': 'bert-base',
                        'selected': True
                    })
                    
                    evaluation['expected_next'] = 'Select BERT Large checkbox'
                
                elif 'bert' in element_name and 'large' in element_name and 'checkbox' in element_name:
                    # BERT Large checkbox clicked
                    evaluation['valid'] = True
                    evaluation['feedback'] = 'BERT Large model selected for comparison'
                    self.test_state['bert_large_selected'] = True
                    
                    # Log the model selection
                    self.validator.log_selected_models('BERT_LARGE_SELECTED', {
                        'model': 'bert-large',
                        'selected': True
                    })
                    
                    if not self.test_state['bert_base_selected']:
                        evaluation['expected_next'] = 'Select BERT Base checkbox'
                    else:
                        evaluation['expected_next'] = 'Click compare button'
                
                elif 'compare' in element_name and 'button' in element_name:
                    # Compare button clicked
                    if self.test_state['bert_base_selected'] and self.test_state['bert_large_selected']:
                        evaluation['valid'] = True
                        evaluation['feedback'] = 'Compare button clicked - generating comparison table'
                        self.test_state['compare_button_clicked'] = True
                        
                        # Log the compare action
                        self.validator.log_selected_models('COMPARE_BUTTON_CLICKED', {
                            'models_to_compare': ['bert-base', 'bert-large'],
                            'comparison_initiated': True
                        })
                        
                        # Wait for comparison table to load
                        await asyncio.sleep(2)
                        updated_models_info = await self._get_current_models_info()
                        
                        if updated_models_info.get('comparison_table_visible'):
                            self.test_state['comparison_table_visible'] = True
                            self.test_state['metrics_displayed'] = True
                            
                            # Log successful comparison
                            self.validator.log_selected_models('COMPARISON_TABLE_DISPLAYED', {
                                'table_visible': True,
                                'metrics_available': True,
                                'comparison_complete': True
                            })
                            
                            evaluation['test_completed'] = True
                            evaluation['validation_result'] = self.validator.validate_model_comparison_result()
                        else:
                            evaluation['feedback'] = 'Compare button clicked but comparison table did not appear'
                    else:
                        evaluation['feedback'] = 'Compare button clicked but both models were not selected first'
                
                else:
                    evaluation['feedback'] = f'Clicked wrong element - expected model checkbox or compare button, got: {element_name}'
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
                if 'bert' in element_name and 'base' in element_name:
                    coords = self.huggingface_elements_config.get('bert_base_checkbox', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.1), coords.get('y', 0.4)
                elif 'bert' in element_name and 'large' in element_name:
                    coords = self.huggingface_elements_config.get('bert_large_checkbox', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.1), coords.get('y', 0.6)
                elif 'compare' in element_name and 'button' in element_name:
                    coords = self.huggingface_elements_config.get('compare_button', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.8), coords.get('y', 0.9)
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
            'bert_base_selected': self.test_state['bert_base_selected'],
            'bert_large_selected': self.test_state['bert_large_selected'],
            'compare_button_clicked': self.test_state['compare_button_clicked'],
            'comparison_table_visible': self.test_state['comparison_table_visible'],
            'test_completed': self.test_state['comparison_table_visible'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'huggingface_model_comparison',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'model_selections': self.validator.get_selection_log(),
            'validation_summary': self.validator.validate_model_comparison_result(),
            'element_configuration': self.huggingface_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_huggingface_model_comparison(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for HuggingFace model comparison with real browser integration.
    """
    test_env = HuggingFaceTestEnvironment(url, backend_url, config_path)
    
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
def test_huggingface_model_comparison_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_huggingface_model_comparison(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_huggingface_model_comparison_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> HuggingFaceTestEnvironment:
    """Factory function to create a HuggingFace model comparison test environment."""
    return HuggingFaceTestEnvironment(url, backend_url, config_path)