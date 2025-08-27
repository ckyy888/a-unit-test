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

# Dummy coordinates for HuggingFace Leaderboard elements
QWEN25_BBOX = {
    "x": 254,
    "y": 1057,
    "width": 479,
    "height": 52,
}

SEARCH_BBOX = {
    "x": 688,
    "y": 19,
    "width": 511,
    "height": 48,
}

MODEL1_BBOX = {
    "x": 1457,
    "y": 19,
    "width": 125,
    "height": 48,
}

MODEL2_BBOX = {
    "x": 862,
    "y": 1177,
    "width": 95,
    "height": 37,
}

class HuggingFaceLeaderboardTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.leaderboard_screenshot_path = os.path.join(self.screenshots_dir, 'llm_leaderboard.png')
        self.deviation_screenshot_path = os.path.join(self.screenshots_dir, 'deviation.png')
    
    def get_leaderboard_screenshot(self):
        try:
            # Open image and return PIL Image object (what agent might expect)
            img = Image.open(self.leaderboard_screenshot_path)
            return img
        except Exception as e:
            print(f"Error loading leaderboard screenshot: {e}")
            # Return a dummy image if file not found
            return Image.new('RGB', (1920, 1080), color='white')
    
    def get_deviation_screenshot(self):
        try:
            # Open image and return PIL Image object (what agent might expect)
            img = Image.open(self.deviation_screenshot_path)
            return img
        except Exception as e:
            print(f"Error loading deviation screenshot: {e}")
            # Return a dummy image if file not found
            return Image.new('RGB', (1920, 1080), color='white')
    
    def is_point_inside_bbox(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        """Check if a point (x, y) is inside the given bounding box"""
        left = bbox["x"]
        right = bbox["x"] + bbox["width"]
        top = bbox["y"]
        bottom = bbox["y"] + bbox["height"]
        return (left <= x <= right and top <= y <= bottom)
    
    def is_click_in_qwen25_bbox(self, x: int, y: int) -> bool:
        """Check if click is inside the Qwen2.5 bounding box"""
        return self.is_point_inside_bbox(x, y, QWEN25_BBOX)
    
    def is_click_in_target_boxes(self, x: int, y: int) -> bool:
        """Check if click is inside search box or either model box"""
        return (self.is_point_inside_bbox(x, y, SEARCH_BBOX) or 
                self.is_point_inside_bbox(x, y, MODEL1_BBOX) or
                self.is_point_inside_bbox(x, y, MODEL2_BBOX))
    
    def evaluate_first_agent_click(self, coordinates: Tuple[int, int]) -> Dict[str, Any]:
        """Evaluate if the first agent clicked on Qwen2.5 bounding box"""
        if coordinates is None:
            return {
                "passed": False,
                "reason": "No click action was performed",
                "expected": "Click on Qwen2.5 bounding box",
                "actual": "No click detected"
            }
        
        x, y = coordinates
        is_inside = self.is_click_in_qwen25_bbox(x, y)
        
        return {
            "passed": is_inside,
            "reason": f"Click at ({x}, {y}) {'inside' if is_inside else 'outside'} Qwen2.5 bounding box",
            "expected": "Click on Qwen2.5 bounding box",
            "actual": f"Click at ({x}, {y})",
            "coordinates": coordinates,
            "qwen25_bbox": QWEN25_BBOX
        }
    
    def evaluate_second_agent_click(self, coordinates: Tuple[int, int]) -> Dict[str, Any]:
        """Evaluate if the second agent clicked on search or model bounding boxes"""
        if coordinates is None:
            return {
                "passed": False,
                "reason": "No click action was performed",
                "expected": "Click on search box or model bounding boxes",
                "actual": "No click detected"
            }
        
        x, y = coordinates
        is_inside = self.is_click_in_target_boxes(x, y)
        
        # Determine which box was clicked
        clicked_box = "unknown"
        if self.is_point_inside_bbox(x, y, SEARCH_BBOX):
            clicked_box = "search"
        elif self.is_point_inside_bbox(x, y, MODEL1_BBOX):
            clicked_box = "model1"
        elif self.is_point_inside_bbox(x, y, MODEL2_BBOX):
            clicked_box = "model2"
        
        return {
            "passed": is_inside,
            "reason": f"Click at ({x}, {y}) {'inside' if is_inside else 'outside'} target bounding boxes ({clicked_box})",
            "expected": "Click on search box or model bounding boxes",
            "actual": f"Click at ({x}, {y}) in {clicked_box} area",
            "coordinates": coordinates,
            "search_bbox": SEARCH_BBOX,
            "model1_bbox": MODEL1_BBOX,
            "model2_bbox": MODEL2_BBOX,
            "clicked_box": clicked_box
        }
    
    async def run_first_agent_test(self, task_description: str) -> Dict[str, Any]:
        """Run test with first agent - should click on Qwen2.5"""
        print("Running first agent test (click on Qwen2.5)...")
        
        # Run the agent with leaderboard screenshot
        result = await run_agent_test(self.get_leaderboard_screenshot, task_description)
        
        # Evaluate the result
        evaluation = self.evaluate_first_agent_click(result.get('last_coordinates'))
        
        return {
            "test_name": "first_agent_test",
            "screenshot_type": "leaderboard",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
    
    async def run_second_agent_test(self, task_description_stage1: str, task_description_stage2: str) -> Dict[str, Any]:
        """Run test with second agent - must first click Qwen2.5, then navigate and click on search/model boxes"""
        print("Running second agent test (two-stage: Qwen2.5 then deviation page)...")
        
        # Stage 1: Agent must click on Qwen2.5 in leaderboard
        print("  Stage 1: Testing Qwen2.5 click on leaderboard...")
        stage1_result = await run_agent_test(self.get_leaderboard_screenshot, task_description_stage1)
        stage1_evaluation = self.evaluate_first_agent_click(stage1_result.get('last_coordinates'))
        
        # If stage 1 fails, the entire test fails
        if not stage1_evaluation["passed"]:
            return {
                "test_name": "second_agent_test", 
                "screenshot_type": "leaderboard",
                "stage": 1,
                "agent_result": stage1_result,
                "evaluation": stage1_evaluation,
                "passed": False,
                "failure_reason": "Failed to click Qwen2.5 in stage 1"
            }
        
        # Stage 2: Agent gets deviation.png and must click search/model boxes
        print("  Stage 2: Testing search/model click on deviation page...")
        stage2_result = await run_agent_test(self.get_deviation_screenshot, task_description_stage2)
        stage2_evaluation = self.evaluate_second_agent_click(stage2_result.get('last_coordinates'))
        
        # Overall success requires both stages to pass
        overall_passed = stage1_evaluation["passed"] and stage2_evaluation["passed"]
        
        return {
            "test_name": "second_agent_test", 
            "screenshot_type": "both",
            "stage1_result": stage1_result,
            "stage1_evaluation": stage1_evaluation,
            "stage2_result": stage2_result,
            "stage2_evaluation": stage2_evaluation,
            "passed": overall_passed,
            "stages_completed": 2 if stage2_evaluation else 1
        }
    
    async def run_both_tests(self, first_task_description: str, second_task_stage1_description: str, second_task_stage2_description: str) -> Dict[str, Any]:
        """Run both test scenarios"""
        print("Starting HuggingFace Leaderboard Test Suite...")
        
        # Run both tests
        first_test_result = await self.run_first_agent_test(first_task_description)
        second_test_result = await self.run_second_agent_test(second_task_stage1_description, second_task_stage2_description)
        
        # Overall results
        overall_passed = first_test_result["passed"] and second_test_result["passed"]
        
        results = {
            "overall_passed": overall_passed,
            "first_agent_test": first_test_result,
            "second_agent_test": second_test_result,
            "summary": {
                "first_test_passed": first_test_result["passed"],
                "second_test_passed": second_test_result["passed"],
                "total_tests": 2,
                "passed_tests": sum([first_test_result["passed"], second_test_result["passed"]])
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
        test_name: str = "test"
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
                    draw.text((gt_x1, gt_y1 - 20), 'Target Box', fill='blue', font=font)  # type: ignore
                else:
                    draw.text((gt_x1, 0), 'Target Box', fill='blue', font=font)  # type: ignore

            if pred_x is not None and pred_y is not None:
                # Different color based on whether it's in bbox or not
                outline_color = 'green' if is_in_bbox else 'red'
                draw.ellipse(
                    [(pred_x - 15, pred_y - 15), (pred_x + 15, pred_y + 15)], outline=outline_color, width=3
                )
                draw.text((pred_x + 20, pred_y + 30), 'Click', fill=outline_color, font=font)

            output_filename = f'{test_name}.png'
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

async def test_huggingface_leaderboard():
    """Main test function that runs both scenarios"""
    first_task_description = """
    You are an AI assistant controlling a computer to interact with a HuggingFace leaderboard page.
    The screen resolution is 2940x1596.
    
    TASK: Click on the "Qwen2.5-72B-Instruct" model in the leaderboard.
    
    INSTRUCTIONS:
    1. First, take a screenshot to see the current page
    2. Look for the "Qwen2.5-72B-Instruct" model entry in the leaderboard
    3. Click on the "Qwen2.5-72B-Instruct" model name or row
    
    You have access to computer control functions including screenshot, click, type, and move.
    Start by taking a screenshot to see what's on the page.
    """
    
    second_task_stage1_description = """
    You are an AI assistant controlling a computer to interact with a HuggingFace leaderboard page.
    The screen resolution is 2940x1596.
    
    TASK: Click on the "Qwen2.5-72B-Instruct" model in the leaderboard to navigate to its details.
    
    INSTRUCTIONS:
    1. First, take a screenshot to see the current page
    2. Look for the "Qwen2.5-72B-Instruct" model entry in the leaderboard
    3. Click on the "Qwen2.5-72B-Instruct" model name or row to navigate to its page
    
    You have access to computer control functions including screenshot, click, type, and move.
    Start by taking a screenshot to see what's on the page.
    """
    
    second_task_stage2_description = """
    You are an AI assistant controlling a computer to interact with a HuggingFace model page.
    The screen resolution is 2940x1596.
    
    TASK: Navigate the model page and interact with search or model elements.
    
    INSTRUCTIONS:
    1. First, take a screenshot to see the current page
    2. Look for either:
       - A search box/field on the page
       - Model-related clickable elements or links
    3. Click on one of these elements:
       - The search box if available
       - Any model-related clickable elements
    
    You have access to computer control functions including screenshot, click, type, and move.
    Start by taking a screenshot to see what's on the page.
    """
    
    test_runner = HuggingFaceLeaderboardTest()
    results = await test_runner.run_both_tests(first_task_description, second_task_stage1_description, second_task_stage2_description)
    
    # Initialize logger and log results to JSON
    logger = TestLogger("huggingface_leaderboard")
    
    # Still generate visualizations but don't print detailed results
    first_test = results["first_agent_test"]
    if first_test['agent_result'].get('last_coordinates'):
        x, y = first_test['agent_result']['last_coordinates']
        test_runner.visualize_prediction(
            image_path=test_runner.leaderboard_screenshot_path,
            pred_x=x,
            pred_y=y,
            run_id=None,
            gt_x1=QWEN25_BBOX['x'],
            gt_y1=QWEN25_BBOX['y'],
            gt_x2=QWEN25_BBOX['x'] + QWEN25_BBOX['width'],
            gt_y2=QWEN25_BBOX['y'] + QWEN25_BBOX['height'],
            is_in_bbox=first_test['evaluation']['passed'],
            test_name="huggingface_first_agent"
        )
    
    # Second Agent Test Results (Two-Stage)
    second_test = results["second_agent_test"]
    if 'stage1_evaluation' in second_test:
        if second_test['stage1_result'].get('last_coordinates'):
            x1, y1 = second_test['stage1_result']['last_coordinates']
            test_runner.visualize_prediction(
                image_path=test_runner.leaderboard_screenshot_path,
                pred_x=x1,
                pred_y=y1,
                run_id=None,
                gt_x1=QWEN25_BBOX['x'],
                gt_y1=QWEN25_BBOX['y'],
                gt_x2=QWEN25_BBOX['x'] + QWEN25_BBOX['width'],
                gt_y2=QWEN25_BBOX['y'] + QWEN25_BBOX['height'],
                is_in_bbox=second_test['stage1_evaluation']['passed'],
                test_name="huggingface_second_agent_stage1"
            )
        
        if 'stage2_evaluation' in second_test:
            if second_test['stage2_result'].get('last_coordinates'):
                x2, y2 = second_test['stage2_result']['last_coordinates']
                test_runner.visualize_prediction(
                    image_path=test_runner.deviation_screenshot_path,
                    pred_x=x2,
                    pred_y=y2,
                    run_id=None,
                    gt_x1=SEARCH_BBOX['x'],
                    gt_y1=SEARCH_BBOX['y'],
                    gt_x2=SEARCH_BBOX['x'] + SEARCH_BBOX['width'],
                    gt_y2=SEARCH_BBOX['y'] + SEARCH_BBOX['height'],
                    is_in_bbox=second_test['stage2_evaluation']['passed'],
                    test_name="huggingface_second_agent_stage2"
                )
    else:
        # Handle case where second test failed in stage 1
        if second_test['agent_result'].get('last_coordinates'):
            x, y = second_test['agent_result']['last_coordinates']
            test_runner.visualize_prediction(
                image_path=test_runner.leaderboard_screenshot_path,
                pred_x=x,
                pred_y=y,
                run_id=None,
                gt_x1=QWEN25_BBOX['x'],
                gt_y1=QWEN25_BBOX['y'],
                gt_x2=QWEN25_BBOX['x'] + QWEN25_BBOX['width'],
                gt_y2=QWEN25_BBOX['y'] + QWEN25_BBOX['height'],
                is_in_bbox=False,
                test_name="huggingface_second_agent_failed"
            )
    
    # Log results to JSON file and print summary
    logger.log_and_print_summary(results)
    
    # Return results instead of using assert
    return results

async def main():
    """Main function to run the tests"""
    # Run the test and return results
    results = await test_huggingface_leaderboard()
    return results

if __name__ == "__main__":
    results = asyncio.run(main())