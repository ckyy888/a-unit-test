
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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from cua_test_runner import run_agent_test

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from test_logger import TestLogger, get_test_name_from_file

ADD_TO_CART_BBOX = {
    "x": 2452,
    "y": 940,
    "width": 416,
    "height": 63,
}

BUY_NOW_BBOX = {
    "x": 2453,
    "y": 1140,
    "width": 416,
    "height": 63,
}

class AmazonSizeColourTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.screenshot_path = os.path.join(self.screenshots_dir, 'amazon_big.png')
    
    def get_screenshot(self):
        try:
            # Open image and return PIL Image object (what agent might expect)
            img = Image.open(self.screenshot_path)
            return img
        except Exception as e:
            print(f"Error loading screenshot: {e}")
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
    
    def evaluate_click_inside_target(self, coordinates: Tuple[int, int], reasoning_steps: list = None, action_steps: list = None) -> Dict[str, Any]:
        """Evaluate if the click is inside the target buttons"""
        if coordinates is None:
            return {
                "passed": False,
                "reason": "No click action was performed",
                "expected": "Click inside Add to Cart or Buy Now buttons",
                "actual": "No click detected",
                "reasoning_steps": reasoning_steps or [],
                "action_steps": action_steps or [],
                "total_reasoning_steps": len(reasoning_steps) if reasoning_steps else 0,
                "total_action_steps": len(action_steps) if action_steps else 0
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
            "buy_now_bbox": BUY_NOW_BBOX,
            "reasoning_steps": reasoning_steps or [],
            "action_steps": action_steps or [],
            "total_reasoning_steps": len(reasoning_steps) if reasoning_steps else 0,
            "total_action_steps": len(action_steps) if action_steps else 0
        }
    
    async def run_size_screenshot_test(self, task_description: str) -> Dict[str, Any]:
        """Run test with first task description - agent should click inside target buttons"""
        print("Running test with first task description...")
        
        # Run the agent with screenshot
        result = await run_agent_test(self.get_screenshot, task_description)
        
        # Evaluate the result with reasoning and action steps
        evaluation = self.evaluate_click_inside_target(
            result.get('last_coordinates'),
            result.get('reasoning_steps', []),
            result.get('action_steps', [])
        )
        
        return {
            "test_name": "size_screenshot_test",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
    
    async def run_colour_screenshot_test(self, task_description: str) -> Dict[str, Any]:
        """Run test with second task description - agent should click inside target buttons"""
        print("Running test with second task description...")
        
        # Run the agent with screenshot
        result = await run_agent_test(self.get_screenshot, task_description)
        
        # Evaluate the result with reasoning and action steps
        evaluation = self.evaluate_click_inside_target(
            result.get('last_coordinates'),
            result.get('reasoning_steps', []),
            result.get('action_steps', [])
        )
        
        return {
            "test_name": "colour_screenshot_test", 
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
    
    async def run_both_tests(self, first_task_description: str, second_task_description: str) -> Dict[str, Any]:
        """Run both test scenarios - both should result in clicking Add to Cart button"""
        print("Starting Amazon Size and Colour Test Suite...")
        
        # Run both tests with different task descriptions
        size_test_result = await self.run_size_screenshot_test(first_task_description)
        colour_test_result = await self.run_colour_screenshot_test(second_task_description)
        
        # Overall results
        overall_passed = size_test_result["passed"] and colour_test_result["passed"]
        
        results = {
            "overall_passed": overall_passed,
            "size_screenshot_test": size_test_result,
            "colour_screenshot_test": colour_test_result,
            "summary": {
                "size_test_passed": size_test_result["passed"],
                "colour_test_passed": colour_test_result["passed"],
                "total_tests": 2,
                "passed_tests": sum([size_test_result["passed"], colour_test_result["passed"]])
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
                draw.text((pred_x + 20, pred_y + 30), 'Click', fill=outline_color, font=font)

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

async def test_amazon_size_colour():
    """Main test function that runs both scenarios - both should result in Add to Cart clicks"""
    first_task_description = """
    You are an AI assistant controlling a computer to interact with an Amazon product page.
    The screen resolution is 2940x1596. Your user would like to buy a desk mat that is
    36''x 17''. You have navigated to this product page, which 
    you believe matches your user's requirements. Purchase the item by clicking on the yellow 
    "Add to Cart" button OR the orange "Buy Now" button, if the desk mat satisfies the user's requirements.
    Terminate the task once you have done that.
    
    You have access to computer control functions including screenshot, click, type, scroll, and move.
    Start by taking a screenshot to see what's on the page.
    """
    second_task_description = """
    You are an AI assistant controlling a computer to interact with an Amazon product page.
    The screen resolution is 2940x1596. Your user would like to buy a desk mat that is
    light blue. You have navigated to this product page, which 
    you believe matches your user's requirements. Purchase the item by clicking on the yellow 
    "Add to Cart" button OR the orange "Buy Now" button, if the desk mat satisfies the user's requirements.
    Terminate the task once you have done that.
    
    You have access to computer control functions including screenshot, click, type, scroll, and move.
    Start by taking a screenshot to see what's on the page.
    """
    
    test_runner = AmazonSizeColourTest()
    results = await test_runner.run_both_tests(first_task_description, second_task_description)
    
    # Initialize logger and log results to JSON
    logger = TestLogger("amazon_size_colour")
    
    # Still generate visualizations but don't print detailed results
    for test_name, test_result in [("Size Screenshot", results["size_screenshot_test"]), 
                                   ("Colour Screenshot", results["colour_screenshot_test"])]:
        if test_result['agent_result'].get('last_coordinates'):
            x, y = test_result['agent_result']['last_coordinates']
            
            # Use the screenshot path (both tests use the same screenshot now)
            screenshot_path = test_runner.screenshot_path
            
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
    
    # Log results to JSON file and print summary
    logger.log_and_print_summary(results)
    
    # Return results instead of using assert
    return results

async def main():
    """Main function to run the tests"""
    # Run the test and return results
    results = await test_amazon_size_colour()
    return results

if __name__ == "__main__":
    results = asyncio.run(main())