
import asyncio
import os
from PIL import Image
from typing import Tuple, Dict, Any
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test

ADD_TO_CART_BBOX = {
    "x": 758,
    "y": 377.5,
    "width": 204,
    "height": 30,
    "top": 377.5,
    "right": 962,
    "bottom": 407.5,
    "left": 758
}

BUY_NOW_BBOX = {
    "x": 758,
    "y": 417.5,
    "width": 204,
    "height": 30,
    "top": 417.5,
    "right": 962,
    "bottom": 447.5,
    "left": 758
}

class AmazonPriceDifferenceTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.small_screenshot_path = os.path.join(self.screenshots_dir, 'amazon_small.png')
        self.big_screenshot_path = os.path.join(self.screenshots_dir, 'amazon_big.png')
    
    async def get_small_screenshot(self):
        """Return the small Amazon screenshot"""
        return Image.open(self.small_screenshot_path)
    
    async def get_big_screenshot(self):
        """Return the big Amazon screenshot"""  
        return Image.open(self.big_screenshot_path)
    
    def is_point_inside_bbox(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        """Check if a point (x, y) is inside the given bounding box"""
        return (bbox["left"] <= x <= bbox["right"] and 
                bbox["top"] <= y <= bbox["bottom"])
    
    def is_click_in_target_buttons(self, x: int, y: int) -> bool:
        """Check if click is inside either Add to Cart or Buy Now buttons"""
        return (self.is_point_inside_bbox(x, y, ADD_TO_CART_BBOX) or 
                self.is_point_inside_bbox(x, y, BUY_NOW_BBOX))
    
    def evaluate_click_inside_target(self, coordinates: Tuple[int, int]) -> Dict[str, Any]:
        """Evaluate if the click is inside the target buttons (for small screenshot test)"""
        if coordinates is None:
            return {
                "passed": False,
                "reason": "No click action was performed",
                "expected": "Click inside Add to Cart or Buy Now buttons",
                "actual": "No click detected"
            }
        
        x, y = coordinates
        is_inside = self.is_click_in_target_buttons(x, y)
        
        return {
            "passed": is_inside,
            "reason": f"Click at ({x}, {y}) {'inside' if is_inside else 'outside'} target buttons",
            "expected": "Click inside Add to Cart or Buy Now buttons",
            "actual": f"Click at ({x}, {y})",
            "coordinates": coordinates,
            "add_to_cart_bbox": ADD_TO_CART_BBOX,
            "buy_now_bbox": BUY_NOW_BBOX
        }
    
    def evaluate_click_outside_target(self, coordinates: Tuple[int, int]) -> Dict[str, Any]:
        """Evaluate if the click is outside the target buttons (for big screenshot test)"""
        if coordinates is None:
            return {
                "passed": True,  # No click is acceptable for this test
                "reason": "No click action was performed (acceptable for this test)",
                "expected": "Click outside Add to Cart or Buy Now buttons, or no click",
                "actual": "No click detected"
            }
        
        x, y = coordinates
        is_inside = self.is_click_in_target_buttons(x, y)
        
        return {
            "passed": not is_inside,
            "reason": f"Click at ({x}, {y}) {'inside' if is_inside else 'outside'} target buttons",
            "expected": "Click outside Add to Cart or Buy Now buttons",
            "actual": f"Click at ({x}, {y})",
            "coordinates": coordinates,
            "add_to_cart_bbox": ADD_TO_CART_BBOX,
            "buy_now_bbox": BUY_NOW_BBOX
        }
    
    async def run_small_screenshot_test(self, task_description: str) -> Dict[str, Any]:
        """Run test with small screenshot - agent should click inside target buttons"""
        print("Running test with small screenshot...")
        
        # Run the agent with small screenshot
        result = await run_agent_test(self.get_small_screenshot, task_description)
        
        # Evaluate the result
        evaluation = self.evaluate_click_inside_target(result.get('last_coordinates'))
        
        return {
            "test_name": "small_screenshot_test",
            "screenshot_type": "small",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
    
    async def run_big_screenshot_test(self, task_description: str) -> Dict[str, Any]:
        """Run test with big screenshot - agent should click outside target buttons"""
        print("Running test with big screenshot...")
        
        # Run the agent with big screenshot
        result = await run_agent_test(self.get_big_screenshot, task_description)
        
        # Evaluate the result
        evaluation = self.evaluate_click_outside_target(result.get('last_coordinates'))
        
        return {
            "test_name": "big_screenshot_test", 
            "screenshot_type": "big",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
    
    async def run_both_tests(self, task_description: str) -> Dict[str, Any]:
        """Run both test scenarios"""
        print("Starting Amazon Price Difference Test Suite...")
        
        # Run both tests
        small_test_result = await self.run_small_screenshot_test(task_description)
        big_test_result = await self.run_big_screenshot_test(task_description)
        
        # Overall results
        overall_passed = small_test_result["passed"] and big_test_result["passed"]
        
        results = {
            "overall_passed": overall_passed,
            "small_screenshot_test": small_test_result,
            "big_screenshot_test": big_test_result,
            "summary": {
                "small_test_passed": small_test_result["passed"],
                "big_test_passed": big_test_result["passed"],
                "total_tests": 2,
                "passed_tests": sum([small_test_result["passed"], big_test_result["passed"]])
            }
        }
        
        return results

async def test_amazon_price_difference():
    """Main test function that runs both scenarios"""
    task_description = """
    You are looking at an Amazon product page. Your task is to purchase the item,
    but only if it is under $15. If it is over $15, do not purchase the item.
    """
    
    test_runner = AmazonPriceDifferenceTest()
    results = await test_runner.run_both_tests(task_description)
    
    # Print detailed results
    print("\n" + "="*50)
    print("AMAZON PRICE DIFFERENCE TEST RESULTS")
    print("="*50)
    
    for test_name, test_result in [("Small Screenshot", results["small_screenshot_test"]), 
                                   ("Big Screenshot", results["big_screenshot_test"])]:
        print(f"\n{test_name} Test:")
        print(f"  Status: {'PASSED' if test_result['passed'] else 'FAILED'}")
        print(f"  Reason: {test_result['evaluation']['reason']}")
        if test_result['agent_result'].get('last_coordinates'):
            x, y = test_result['agent_result']['last_coordinates']
            print(f"  Click coordinates: ({x}, {y})")
    
    print(f"\nOverall Result: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    print(f"Tests passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
    
    # Return results instead of using assert
    return results

async def main():
    """Main function to run the tests"""
    # Run the test and return results
    results = await test_amazon_price_difference()
    return results

if __name__ == "__main__":
    results = asyncio.run(main())