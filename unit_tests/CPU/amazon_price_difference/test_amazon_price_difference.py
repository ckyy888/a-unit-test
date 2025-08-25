
import asyncio
import os
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, Any
import sys
from dotenv import load_dotenv
import urllib.parse
from typing import Any, Dict, Optional

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test

ADD_TO_CART_BBOX = {
    "x": 2452,
    "y": 650,
    "width": 416,
    "height": 63,
}

BUY_NOW_BBOX = {
    "x": 2453,
    "y": 730,
    "width": 416,
    "height": 63,
}

class AmazonPriceDifferenceTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.small_screenshot_path = os.path.join(self.screenshots_dir, 'amazon_small.png')
        self.big_screenshot_path = os.path.join(self.screenshots_dir, 'amazon_big.png')
    
    def get_small_screenshot(self):
        try:
            # Open image and return PIL Image object (what agent might expect)
            img = Image.open(self.small_screenshot_path)
            return img
        except Exception as e:
            print(f"Error loading small screenshot: {e}")
            # Return a dummy image if file not found
            return Image.new('RGB', (1920, 1080), color='white')
    
    def get_big_screenshot(self):
        try:
            # Open image and return PIL Image object (what agent might expect)
            img = Image.open(self.big_screenshot_path)
            return img
        except Exception as e:
            print(f"Error loading big screenshot: {e}")
            # Return a dummy image if file not found
            return Image.new('RGB', (1920, 1080), color='white')
    
    def is_point_inside_bbox(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        """Check if a point (x, y) is inside the given bounding box"""
        left = bbox["x"]
        right = bbox["x"] + bbox["width"]
        top = bbox["y"]
        bottom = bbox["y"] + bbox["height"]
        return (left <= x <= right and top <= y <= bottom)
    
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
    
    def visualize_prediction(
        self,
        image_path: str,
        pred_x: Optional[int],
        pred_y: Optional[int],
        run_id: Optional[str],
        gt_x1: Optional[int],
        gt_y1: Optional[int],
        gt_x2: Optional[int],
        gt_y2: Optional[int],
        is_in_bbox: Optional[bool] = None,
    ) -> Optional[str]:
        try:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            if run_id:
                vis_dir = os.path.join(base_dir, 'results', run_id, 'visualizations')
            else:
                vis_dir = os.path.join(base_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            print(f'Visualization directory: {vis_dir}')

            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype('Arial.ttf', 16)
            except IOError:
                font = ImageFont.load_default()

            image_filename = os.path.basename(image_path)

            # Draw the bounding box if available
            if all(v is not None for v in [gt_x1, gt_y1, gt_x2, gt_y2]):
                draw.rectangle([(gt_x1, gt_y1), (gt_x2, gt_y2)], outline='blue', width=2)  # type: ignore
                if gt_y1 is not None:
                    draw.text((gt_x1, gt_y1 - 20), 'Bounding Box', fill='blue', font=font)  # type: ignore
                else:
                    draw.text((gt_x1, 0), 'Bounding Box', fill='blue', font=font)  # type: ignore

            if pred_x is not None and pred_y is not None:
                # Different color based on whether it's in bbox or not
                outline_color = 'orange' if is_in_bbox else 'red'
                draw.ellipse(
                    [(pred_x - 15, pred_y - 15), (pred_x + 15, pred_y + 15)], outline=outline_color, width=3
                )
                draw.text((pred_x + 20, pred_y + 30), 'Prediction', fill=outline_color, font=font)

            output_filename = f'rec1.png'
            if is_in_bbox:
                output_dir = os.path.join(vis_dir, 'correct')
            else:
                output_dir = os.path.join(vis_dir, 'incorrect')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            img.save(output_path)

            return output_path
        except Exception as e:
            print(f'Error creating visualization: {e}')
            return None

async def test_amazon_price_difference():
    """Main test function that runs both scenarios"""
    task_description = """
    You are an AI assistant controlling a computer to interact with an Amazon product page.
    The screen resolution is 2940x1596.
    
    TASK: Purchase the item ONLY if the price is under $15.
    
    INSTRUCTIONS:
    1. First, take a screenshot to see the current page
    2. Look for the price of the item on the page (usually shown as $XX.XX)
    3. If the price is under $15.00:
       - Click on the yellow "Add to Cart" button OR the orange "Buy Now" button
    4. If the price is $15.00 or higher:
       - Do NOT click on any purchase buttons
       - Do NOT take any action
    
    You have access to computer control functions including screenshot, click, type, scroll, and move.
    Start by taking a screenshot to see what's on the page.
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
            
            # Use the appropriate screenshot based on test type
            screenshot_type = test_result['screenshot_type']
            if screenshot_type == 'small':
                screenshot_path = test_runner.small_screenshot_path
            else:
                screenshot_path = test_runner.big_screenshot_path
            
            test_runner.visualize_prediction(
                image_path=screenshot_path,
                pred_x=x,
                pred_y=y,
                run_id=None,
                gt_x1=ADD_TO_CART_BBOX['x'],
                gt_y1=ADD_TO_CART_BBOX['y'],
                gt_x2=ADD_TO_CART_BBOX['x'] + ADD_TO_CART_BBOX['width'],
                gt_y2=ADD_TO_CART_BBOX['y'] + ADD_TO_CART_BBOX['height'],
                is_in_bbox=test_result['evaluation']['passed']
            )
    
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