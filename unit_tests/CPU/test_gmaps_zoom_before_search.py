"""
Google Maps Zoom Before Search Test Framework
Purpose: Test agent's ability to zoom in to distinguish clustered café markers before selection
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


class GMapsZoomValidator:
    """Validator for Google Maps zoom and marker selection operations."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.zoom_actions = []
        self.zoom_level_changes = 0
        self.marker_selected = False
        self.cafe_details_opened = False
        
        if config_loader:
            self.config_loader = config_loader
        else:
            self.config_loader = ConfigLoader()
    
    def log_zoom_level_and_marker_selection(self, action_type: str, zoom_data: Dict) -> None:
        """Log zoom level changes and marker selection."""
        zoom_log = {
            'action_id': len(self.zoom_actions) + 1,
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'zoom_data': zoom_data
        }
        
        self.zoom_actions.append(zoom_log)
        self.logger.info(f"Zoom action: {action_type}")
        
        if action_type == 'ZOOM_IN':
            self.zoom_level_changes += 1
            print(f"Zoom level and marker selection: Zoom level increased to {zoom_data.get('zoom_level', 'higher level')}")
        elif action_type == 'MARKER_SELECTED':
            self.marker_selected = True
            print(f"Zoom level and marker selection: Specific café marker selected - {zoom_data.get('cafe_name', 'individual café')}")
        elif action_type == 'DETAILS_PANEL_OPENED':
            self.cafe_details_opened = True
            print(f"Zoom level and marker selection: Café details panel opened correctly")
    
    def validate_zoom_search_result(self) -> Dict[str, Any]:
        """Validate the complete zoom and search sequence."""
        validation_result = {
            'zoom_level_changes': self.zoom_level_changes,
            'sufficient_zoom': self.zoom_level_changes >= 2,
            'marker_selected': self.marker_selected,
            'cafe_details_opened': self.cafe_details_opened,
            'search_successful': False,
            'errors': [],
            'warnings': []
        }
        
        # Check if sufficient zoom was performed (2 clicks)
        if self.zoom_level_changes < 2:
            validation_result['errors'].append(f"Need 2 zoom-in actions, got {self.zoom_level_changes}")
        
        # Check if marker was selected
        if not self.marker_selected:
            validation_result['errors'].append("Specific café marker was not selected")
        
        # Check if details panel opened
        if not self.cafe_details_opened:
            validation_result['errors'].append("Café details panel did not open")
        
        # Determine if search was successful
        validation_result['search_successful'] = (
            self.zoom_level_changes >= 2 and 
            self.marker_selected and 
            self.cafe_details_opened
        )
        
        # Overall validation status
        validation_result['overall_valid'] = (
            validation_result['search_successful'] and
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    def get_zoom_log(self) -> List[Dict]:
        """Return the complete log of zoom actions."""
        return self.zoom_actions


class GMapsZoomTestEnvironment:
    """Test environment for Google Maps zoom before search with browser-in-browser integration."""
    
    def __init__(self, url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None):
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        
        self.url = url
        # Use config for backend URL if not provided
        self.backend_url = backend_url or self.config_loader.get('browser_settings.backend_url', 'http://localhost:8000')
        
        # Initialize validator with config
        self.validator = GMapsZoomValidator(self.config_loader)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration values
        self.max_steps = 4  # Specific to this test
        self.timeout_seconds = self.config_loader.get('test_configuration.timeout_seconds', 30)
        self.screenshot_enabled = self.config_loader.get('test_configuration.screenshot_enabled', True)
        self.gmaps_elements_config = self.config_loader.get_section('gmaps_zoom_elements')
        
        self.test_state = {
            'current_step': 0,
            'zoom_level': 15,  # Starting zoom level
            'markers_clustered': True,
            'zoom_clicks': 0,
            'marker_clicked': False,
            'details_panel_visible': False,
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
        """Get initial test setup with clustered café results."""
        try:
            # Initialize browser session
            await self._init_browser_session()
            
            # Navigate to the URL (should show clustered café search results)
            await self._navigate_to_url()
            
            # Take initial screenshot
            screenshot_data = await self._get_screenshot()
            
            # Detect map elements and marker state
            element_detection = await self._detect_map_elements()
            map_state = await self._get_current_map_state()
            
            # Store screenshot
            self.test_state['screenshots'].append({
                'step': 'initial_clustered_results',
                'timestamp': datetime.now().isoformat(),
                'data': screenshot_data
            })
            
            setup_result = {
                'success': True,
                'url': self.url,
                'task_description': 'Zoom in twice to distinguish clustered café markers, then select specific café',
                'element_detection': element_detection,
                'map_state': map_state,
                'initial_screenshot': screenshot_data,
                'map_elements': self.gmaps_elements_config,
                'validation_rules': {
                    'max_steps': self.max_steps,
                    'required_sequence': ['CLICK(zoom_in)', 'CLICK(zoom_in)', 'CLICK(specific_marker)'],
                    'expected_final_state': 'individual café details panel opens correctly'
                },
                'history': [
                    'Google Maps loaded with search query "coffee near me"',
                    'Multiple café results are displayed on map',
                    'Results are clustered together (overlapping markers)',
                    'Markers are too close to distinguish individual cafés',
                    'Need to zoom in to separate markers for selection'
                ],
                'current_map': {
                    'zoom_level': self.test_state['zoom_level'],
                    'markers_clustered': True,
                    'individual_markers_selectable': False
                },
                'current_step': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update test state based on detection
            if element_detection.get('zoom_in', {}).get('found'):
                self.test_state['zoom_controls_available'] = True
            if element_detection.get('cafe_marker', {}).get('found'):
                self.test_state['markers_present'] = True
            
            self.logger.info(f"Google Maps zoom test initialized: {self.url}")
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
    
    async def _detect_map_elements(self) -> Dict:
        """Detect map controls and marker elements on the current page."""
        detected_elements = {}
        
        for element_type, config in self.gmaps_elements_config.items():
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
                                aria_label: element.getAttribute('aria-label') || '',
                                title: element.title || '',
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
    
    async def _get_current_map_state(self) -> Dict:
        """Get current map zoom level and marker state."""
        try:
            eval_script = """
            () => {
                const result = {
                    zoom_level: null,
                    markers_count: 0,
                    markers_clustered: false,
                    details_panel_visible: false
                };
                
                // Try to get zoom level from Google Maps API if available
                if (window.google && window.google.maps) {
                    // Maps API methods to get zoom would go here
                    result.zoom_level = 'available';
                }
                
                // Count visible markers
                const markers = document.querySelectorAll('[data-marker-type="cafe"], .place-marker, [role="button"][aria-label*="coffee"]');
                result.markers_count = markers.length;
                
                // Check for clustering indicators
                const clusterMarkers = document.querySelectorAll('.cluster-marker, [aria-label*="cluster"]');
                result.markers_clustered = clusterMarkers.length > 0;
                
                // Check for details panel
                const detailsPanel = document.querySelector('.place-details-pane, .sidebar, [data-testid="place-details"]');
                if (detailsPanel) {
                    const rect = detailsPanel.getBoundingClientRect();
                    result.details_panel_visible = rect.width > 0 && rect.height > 0;
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
            self.logger.error(f"Failed to get map state: {e}")
            return {}
    
    async def evaluate_agent_action(self, action: Dict) -> Dict:
        """Execute and evaluate agent's zoom and marker selection action."""
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
            
            # Get updated map state
            map_state = await self._get_current_map_state()
            
            # Evaluate the action
            action_type = action.get('action_type')
            element_name = action.get('element_name', '').lower()
            
            if action_type == 6:  # CLICK action
                if 'zoom' in element_name and 'in' in element_name:
                    # Zoom in action
                    evaluation['valid'] = True
                    evaluation['feedback'] = f'Zoom in executed (click {self.test_state["zoom_clicks"] + 1}/2)'
                    self.test_state['zoom_clicks'] += 1
                    self.test_state['zoom_level'] += 1
                    
                    # Log the zoom action
                    self.validator.log_zoom_level_and_marker_selection('ZOOM_IN', {
                        'zoom_level': self.test_state['zoom_level'],
                        'zoom_click_count': self.test_state['zoom_clicks']
                    })
                    
                    if self.test_state['zoom_clicks'] < 2:
                        evaluation['expected_next'] = 'Continue zooming in (need 2 total zoom clicks)'
                    else:
                        evaluation['expected_next'] = 'Select specific café marker'
                        self.test_state['markers_clustered'] = False
                
                elif 'marker' in element_name or 'cafe' in element_name or 'coffee' in element_name:
                    # Café marker selection
                    if self.test_state['zoom_clicks'] >= 2:
                        evaluation['valid'] = True
                        evaluation['feedback'] = 'Specific café marker selected successfully'
                        self.test_state['marker_clicked'] = True
                        
                        # Log the marker selection
                        self.validator.log_zoom_level_and_marker_selection('MARKER_SELECTED', {
                            'cafe_name': 'Individual café marker',
                            'zoom_level': self.test_state['zoom_level']
                        })
                        
                        # Check if details panel opens
                        await asyncio.sleep(2)  # Wait for details panel to load
                        updated_map_state = await self._get_current_map_state()
                        
                        if updated_map_state.get('details_panel_visible'):
                            self.test_state['details_panel_visible'] = True
                            
                            # Log successful details panel opening
                            self.validator.log_zoom_level_and_marker_selection('DETAILS_PANEL_OPENED', {
                                'panel_opened': True,
                                'cafe_details_available': True
                            })
                            
                            evaluation['test_completed'] = True
                            evaluation['validation_result'] = self.validator.validate_zoom_search_result()
                        else:
                            evaluation['feedback'] = 'Marker clicked but details panel did not open'
                    else:
                        evaluation['feedback'] = f'Marker clicked too early - need to zoom in first (current zooms: {self.test_state["zoom_clicks"]}/2)'
                else:
                    evaluation['feedback'] = f'Clicked wrong element - expected zoom control or café marker, got: {element_name}'
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
                if 'zoom' in element_name and 'in' in element_name:
                    coords = self.gmaps_elements_config.get('zoom_in', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.95), coords.get('y', 0.3)
                elif 'marker' in element_name or 'cafe' in element_name:
                    coords = self.gmaps_elements_config.get('cafe_marker', {}).get('coordinates', {})
                    x, y = coords.get('x', 0.6), coords.get('y', 0.5)
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
            'zoom_level': self.test_state['zoom_level'],
            'zoom_clicks': self.test_state['zoom_clicks'],
            'markers_clustered': self.test_state['markers_clustered'],
            'marker_clicked': self.test_state['marker_clicked'],
            'details_panel_visible': self.test_state['details_panel_visible'],
            'test_completed': self.test_state['details_panel_visible'],
            'action_history': self.test_state['action_history'],
            'steps_remaining': self.max_steps - self.test_state['current_step']
        }
    
    def get_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        return {
            'test_type': 'gmaps_zoom_before_search',
            'test_url': self.url,
            'max_steps_limit': self.max_steps,
            'zoom_actions': self.validator.get_zoom_log(),
            'validation_summary': self.validator.validate_zoom_search_result(),
            'element_configuration': self.gmaps_elements_config,
            'timestamp': datetime.now().isoformat()
        }


# Test runner function for external agent systems
async def test_gmaps_zoom_before_search(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """
    Test function for Google Maps zoom before search with real browser integration.
    """
    test_env = GMapsZoomTestEnvironment(url, backend_url, config_path)
    
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
def test_gmaps_zoom_before_search_sync(url: str, run_agent_func: Callable, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> Dict:
    """Synchronous wrapper for the async test function."""
    return asyncio.run(test_gmaps_zoom_before_search(url, run_agent_func, backend_url, config_path))


# Convenience function for quick integration
def create_gmaps_zoom_before_search_test(url: str, backend_url: Optional[str] = None, config_path: Optional[str] = None) -> GMapsZoomTestEnvironment:
    """Factory function to create a Google Maps zoom before search test environment."""
    return GMapsZoomTestEnvironment(url, backend_url, config_path)