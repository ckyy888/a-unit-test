"""
Unified Test Runner for All CPU Test Framework Tests
Purpose: Demonstrate usage of all implemented test frameworks with mock agents
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Callable, Any
from pathlib import Path

# Import all test functions
from test_booking_room_capacity_logic import test_booking_room_capacity_logic_sync
from test_amazon_variant_selection_logic import test_amazon_variant_selection_logic_sync  
from test_wolframalpha_error_recovery import test_wolframalpha_error_recovery_sync
from test_bbc_article_pagination import test_bbc_article_pagination_sync
from test_gmaps_zoom_before_search import test_gmaps_zoom_before_search_sync
from test_allrecipes_sort_then_filter import test_allrecipes_sort_then_filter_sync

# Browser action imports
import sys
sys.path.append('/home/ubuntu/webarena')
from browser_env.actions import (
    create_click_action, 
    create_keyboard_type_action, 
    create_go_back_action,
    create_scroll_action,
    ActionTypes
)


class MockAgentFactory:
    """Factory for creating mock agents for different test scenarios."""
    
    @staticmethod
    def create_booking_capacity_agent() -> Callable:
        """Mock agent for booking room capacity logic test."""
        def mock_agent(initial_observation: Dict) -> List[Dict]:
            return [
                create_click_action(
                    element_role='button',
                    element_name='adult_decrement'
                ),
                create_click_action(
                    element_role='button', 
                    element_name='adult_decrement'
                )
            ]
        return mock_agent
    
    @staticmethod
    def create_amazon_variant_agent() -> Callable:
        """Mock agent for Amazon variant selection test."""
        def mock_agent(initial_observation: Dict) -> List[Dict]:
            return [
                create_click_action(
                    element_role='button',
                    element_name='13 inch size'
                ),
                create_click_action(
                    element_role='button',
                    element_name='available color'
                )
            ]
        return mock_agent
    
    @staticmethod
    def create_wolframalpha_error_agent() -> Callable:
        """Mock agent for WolframAlpha error recovery test."""
        def mock_agent(initial_observation: Dict) -> List[Dict]:
            return [
                create_click_action(
                    element_role='textbox',
                    element_name='query box'
                ),
                create_keyboard_type_action(text='clear'),
                create_keyboard_type_action(text='2+2*3')
            ]
        return mock_agent
    
    @staticmethod
    def create_bbc_pagination_agent() -> Callable:
        """Mock agent for BBC article pagination test."""
        def mock_agent(initial_observation: Dict) -> List[Dict]:
            return [
                create_scroll_action(coordinate=[0, 500]),  # Scroll down
                create_click_action(
                    element_role='link',
                    element_name='next page'
                )
            ]
        return mock_agent
    
    @staticmethod
    def create_gmaps_zoom_agent() -> Callable:
        """Mock agent for Google Maps zoom test."""
        def mock_agent(initial_observation: Dict) -> List[Dict]:
            return [
                create_click_action(
                    element_role='button',
                    element_name='zoom in'
                ),
                create_click_action(
                    element_role='button',
                    element_name='zoom in'
                ),
                create_click_action(
                    element_role='button',
                    element_name='cafe marker'
                )
            ]
        return mock_agent
    
    @staticmethod
    def create_allrecipes_sort_agent() -> Callable:
        """Mock agent for AllRecipes sort and filter test.""" 
        def mock_agent(initial_observation: Dict) -> List[Dict]:
            return [
                create_click_action(
                    element_role='combobox',
                    element_name='sort dropdown'
                ),
                create_click_action(
                    element_role='option', 
                    element_name='rating option'
                ),
                create_click_action(
                    element_role='checkbox',
                    element_name='time filter'
                )
            ]
        return mock_agent


class TestSuiteRunner:
    """Main test suite runner for all CPU framework tests."""
    
    def __init__(self, config_path: str = None, backend_url: str = None):
        self.config_path = config_path or str(Path(__file__).parent / "config.json")
        self.backend_url = backend_url or "http://localhost:8000"
        
        # Load test URLs from config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self.test_urls = self.config.get('test_urls', {})
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize results tracking
        self.test_results = {}
        self.overall_summary = {
            'tests_run': 0,
            'tests_passed': 0, 
            'tests_failed': 0,
            'start_time': None,
            'end_time': None,
            'duration': None
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all implemented tests with their respective mock agents."""
        self.overall_summary['start_time'] = datetime.now().isoformat()
        self.logger.info("Starting CPU Test Framework - All Tests Suite")
        
        # Define test configurations
        test_configs = [
            {
                'name': 'booking_room_capacity_logic',
                'test_func': test_booking_room_capacity_logic_sync,
                'agent_func': MockAgentFactory.create_booking_capacity_agent(),
                'url_key': 'booking_property_page',
                'description': 'Test room capacity management - reduce adult count to resolve capacity warning'
            },
            {
                'name': 'amazon_variant_selection_logic', 
                'test_func': test_amazon_variant_selection_logic_sync,
                'agent_func': MockAgentFactory.create_amazon_variant_agent(),
                'url_key': 'amazon_product_variants',
                'description': 'Test product variant selection - find valid size/color combination'
            },
            {
                'name': 'wolframalpha_error_recovery',
                'test_func': test_wolframalpha_error_recovery_sync,
                'agent_func': MockAgentFactory.create_wolframalpha_error_agent(), 
                'url_key': 'wolframalpha_query_error',
                'description': 'Test error recovery - fix malformed equation syntax'
            },
            {
                'name': 'bbc_article_pagination',
                'test_func': test_bbc_article_pagination_sync,
                'agent_func': MockAgentFactory.create_bbc_pagination_agent(),
                'url_key': 'bbc_article_pagination', 
                'description': 'Test article pagination - navigate to page 2 of multi-page article'
            },
            {
                'name': 'gmaps_zoom_before_search',
                'test_func': test_gmaps_zoom_before_search_sync,
                'agent_func': MockAgentFactory.create_gmaps_zoom_agent(),
                'url_key': 'gmaps_cafe_search',
                'description': 'Test map zoom interaction - zoom in to distinguish clustered markers'
            },
            {
                'name': 'allrecipes_sort_then_filter',
                'test_func': test_allrecipes_sort_then_filter_sync,
                'agent_func': MockAgentFactory.create_allrecipes_sort_agent(),
                'url_key': 'allrecipes_search',
                'description': 'Test sort and filter sequence - sort by rating then apply time filter'
            }
        ]
        
        # Run each test
        for test_config in test_configs:
            self._run_single_test(test_config)
        
        self.overall_summary['end_time'] = datetime.now().isoformat()
        self._calculate_duration()
        self._generate_final_report()
        
        return {
            'overall_summary': self.overall_summary,
            'individual_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _run_single_test(self, test_config: Dict) -> None:
        """Run a single test with error handling."""
        test_name = test_config['name']
        url = self.test_urls.get(test_config['url_key'])
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running Test: {test_name}")
        self.logger.info(f"Description: {test_config['description']}")
        self.logger.info(f"URL: {url}")
        self.logger.info(f"{'='*60}")
        
        self.overall_summary['tests_run'] += 1
        
        try:
            # Run the test
            result = test_config['test_func'](
                url=url,
                run_agent_func=test_config['agent_func'],
                backend_url=self.backend_url,
                config_path=self.config_path
            )
            
            # Store result
            self.test_results[test_name] = {
                'success': result.get('success', False),
                'result': result,
                'config': test_config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update summary
            if result.get('success'):
                self.overall_summary['tests_passed'] += 1
                self.logger.info(f"✅ {test_name} - PASSED")
            else:
                self.overall_summary['tests_failed'] += 1
                self.logger.error(f"❌ {test_name} - FAILED: {result.get('error', 'Unknown error')}")
            
            # Log detailed results
            self._log_test_details(test_name, result)
            
        except Exception as e:
            self.overall_summary['tests_failed'] += 1
            self.logger.error(f"💥 {test_name} - EXCEPTION: {str(e)}")
            
            self.test_results[test_name] = {
                'success': False,
                'result': {'error': str(e), 'exception': True},
                'config': test_config,
                'timestamp': datetime.now().isoformat()
            }
    
    def _log_test_details(self, test_name: str, result: Dict) -> None:
        """Log detailed test results."""
        if result.get('success'):
            # Log successful validation details
            test_report = result.get('test_report', {})
            validation_summary = test_report.get('validation_summary', {})
            
            self.logger.info(f"📊 {test_name} Validation Summary:")
            for key, value in validation_summary.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    self.logger.info(f"  {status} {key}: {value}")
                elif isinstance(value, (int, str)):
                    self.logger.info(f"  📈 {key}: {value}")
        else:
            # Log failure details
            error = result.get('error', 'No error details available')
            self.logger.error(f"🔍 {test_name} Failure Details: {error}")
    
    def _calculate_duration(self) -> None:
        """Calculate test suite duration."""
        if self.overall_summary['start_time'] and self.overall_summary['end_time']:
            start = datetime.fromisoformat(self.overall_summary['start_time'])
            end = datetime.fromisoformat(self.overall_summary['end_time'])
            duration = end - start
            self.overall_summary['duration'] = str(duration)
    
    def _generate_final_report(self) -> None:
        """Generate and log final test suite report."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("CPU TEST FRAMEWORK - FINAL REPORT")
        self.logger.info(f"{'='*80}")
        
        # Overall statistics
        total = self.overall_summary['tests_run']
        passed = self.overall_summary['tests_passed']
        failed = self.overall_summary['tests_failed']
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        self.logger.info(f"📊 Overall Results:")
        self.logger.info(f"  Tests Run: {total}")
        self.logger.info(f"  Passed: {passed}")
        self.logger.info(f"  Failed: {failed}")
        self.logger.info(f"  Pass Rate: {pass_rate:.1f}%")
        self.logger.info(f"  Duration: {self.overall_summary['duration']}")
        
        # Individual test results
        self.logger.info(f"\n📋 Individual Test Results:")
        for test_name, test_data in self.test_results.items():
            status = "✅ PASSED" if test_data['success'] else "❌ FAILED"
            description = test_data['config']['description']
            self.logger.info(f"  {status} {test_name}")
            self.logger.info(f"    Description: {description}")
        
        # Framework capabilities demonstrated
        self.logger.info(f"\n🚀 Framework Capabilities Demonstrated:")
        capabilities = [
            "✅ Real browser automation via Browser-in-Browser integration",
            "✅ Configuration-driven test setup and element detection", 
            "✅ Step-limited agent interaction validation (3-4 steps max)",
            "✅ Detailed action sequence logging and feedback",
            "✅ Multi-platform web application testing (Booking, Amazon, etc.)",
            "✅ Complex user interaction patterns (zoom, sort, filter, paginate)",
            "✅ Error recovery and input validation scenarios",
            "✅ Synchronous and asynchronous test execution patterns",
            "✅ Comprehensive validation with detailed reporting",
            "✅ Modular test design for easy extension and reuse"
        ]
        
        for capability in capabilities:
            self.logger.info(f"  {capability}")
        
        self.logger.info(f"\n{'='*80}")
        
        # Save detailed results to file
        self._save_results_to_file()
    
    def _save_results_to_file(self) -> None:
        """Save detailed test results to JSON file."""
        results_file = Path(__file__).parent / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        full_results = {
            'overall_summary': self.overall_summary,
            'individual_results': self.test_results,
            'framework_info': {
                'version': '1.0.0',
                'test_count': len(self.test_results),
                'browser_integration': True,
                'config_driven': True,
                'agent_agnostic': True
            },
            'generated_at': datetime.now().isoformat()
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            self.logger.info(f"📁 Detailed results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results file: {e}")


def main():
    """Main entry point for running all tests."""
    print("🚀 Starting CPU Test Framework - All Tests Runner")
    print("="*60)
    
    # Initialize and run test suite
    runner = TestSuiteRunner()
    results = runner.run_all_tests()
    
    # Print summary
    print(f"\n🎯 Test Suite Complete!")
    print(f"Pass Rate: {results['overall_summary']['tests_passed']}/{results['overall_summary']['tests_run']} tests")
    
    return results


if __name__ == "__main__":
    main()