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
from cua_test_runner_v2 import run_agent_test
from playwright.async_api import async_playwright  # type: ignore
from test_logger import TestLogger, get_test_name_from_file 

class AllRecipesApplePieTest:
    def __init__(self):
        self.url = "https://www.allrecipes.com/recipe/12682/apple-pie-by-grandma-ople/"
        self.first_task_description = """
        You are a recipe assistant. Your client wants an apple pie recipe that is above 4.0 stars. 
        You are on the All Recipes website and have successfully found the recipe for an apple pie by 
        Grandma Ople. You must verify that the recipe satisfies your client's requirements. If it does, 
        you must save the recipe.
        """
        self.second_task_description = """
        You are a recipe assistant. Your client wants an apple pie recipe that has more 5 star review
        alone than all other review combined. 
        You are on the All Recipes website and have successfully found the recipe for an apple pie by 
        Grandma Ople. You must verify that the recipe satisfies your client's requirements. If it does, 
        you must save the recipe.
        """

    async def _get_save_button_bbox(self) -> Optional[Dict[str, float]]:
        """Fetch the absolute bounding box of the save button in the same viewport as the agent.

        Returns a dict with keys x, y, width, height in pixels, or None if not found.
        """
        # Match the viewport used by PlaywrightComputerHandler (2940x1596)
        viewport_width, viewport_height = 2940, 1596
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor",
                    ],
                )
                context = await browser.new_context(
                    viewport={"width": viewport_width, "height": viewport_height},
                    ignore_https_errors=True,
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                )
                page = await context.new_page()
                await page.goto(self.url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_load_state("networkidle", timeout=30000)

                # Look for save button - try multiple selectors that might contain "Save"
                selectors = [
                    "button:has-text('Save')",
                    "span:has-text('Save')",
                    "[data-testid*='save']",
                    ".save-button",
                    "#save-button",
                    "button[aria-label*='save']",
                    "button[title*='save']"
                ]
                
                element = None
                for selector in selectors:
                    try:
                        await page.wait_for_selector(selector, state="attached", timeout=5000)
                        element = await page.query_selector(selector)
                        if element:
                            break
                    except:
                        continue
                
                if not element:
                    await context.close()
                    await browser.close()
                    return None
                    
                bbox = await element.bounding_box()
                await context.close()
                await browser.close()
                return bbox  # type: ignore[return-value]
        except Exception:
            return None

    def _is_inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return (
            bbox["x"] <= x <= bbox["x"] + bbox["width"]
            and bbox["y"] <= y <= bbox["y"] + bbox["height"]
        )

    async def evaluate_should_save_recipe(self, agent_result: Dict[str, Any], test_scenario: str) -> Dict[str, Any]:
        """Evaluate when agent SHOULD click the save button."""
        coordinates: Optional[Tuple[int, int]] = agent_result.get("last_coordinates")
        
        # Both tests expect the recipe to be saved
        if test_scenario == "first_test":
            expected_message = "Click inside Save button (rating above 4.0 stars)"
        else:
            expected_message = "Click inside Save button (5-star reviews are majority)"
        
        if not coordinates:
            return {
                "passed": False,
                "reason": "No click action was performed",
                "expected": expected_message,
                "actual": "No click detected",
            }

        bbox = await self._get_save_button_bbox()
        if not bbox:
            return {
                "passed": False,
                "reason": "Could not locate Save button by selector",
                "expected": expected_message,
                "actual": f"Coordinates: {coordinates}",
            }

        x, y = coordinates
        inside = self._is_inside(x, y, bbox)
        return {
            "passed": inside,
            "reason": f"Click at ({x}, {y}) {'inside' if inside else 'outside'} Save button",
            "expected": expected_message,
            "actual": f"Click at ({x}, {y})",
            "coordinates": coordinates,
            "save_button_bbox": bbox,
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        """Run first test - agent should save the recipe (rating above 4.0 stars)"""
        print("Running first test (should save recipe - above 4.0 stars)...")
        
        # Run the agent with the URL
        result = await run_agent_test(self.url, task_description)
        
        # Evaluate the result - expect save button to be clicked
        evaluation = await self.evaluate_should_save_recipe(result, "first_test")
        
        return {
            "test_name": "first_test",
            "scenario": "should_save_recipe",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
        
    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        """Run second test - agent should save the recipe (5-star reviews are majority)"""
        print("Running second test (should save recipe - 5-star reviews are majority)...")
        
        # Run the agent with the URL
        result = await run_agent_test(self.url, task_description)
        
        # Evaluate the result - expect save button to be clicked
        evaluation = await self.evaluate_should_save_recipe(result, "second_test")
        
        return {
            "test_name": "second_test", 
            "scenario": "should_save_recipe",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
    
    async def run_both_tests(self, first_task_description: str, second_task_description: str) -> Dict[str, Any]:
        """Run both test scenarios"""
        print("Starting All Recipes Apple Pie Test Suite...")
        
        # Run both tests with their respective task descriptions
        first_test_result = await self.run_first_test(first_task_description)
        second_test_result = await self.run_second_test(second_task_description)
        
        # Overall results
        overall_passed = first_test_result["passed"] and second_test_result["passed"]
        
        results = {
            "overall_passed": overall_passed,
            "first_test": first_test_result,
            "second_test": second_test_result,
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
    ) -> Optional[str]:
        try:
            # Use current working directory instead of navigating up directories
            current_dir = os.getcwd()
            if run_id:
                vis_dir = os.path.join(current_dir, 'visualizations', run_id)
            else:
                vis_dir = os.path.join(current_dir, 'visualizations')
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

async def test_allrecipes_apple_pie():
    """Main test function that runs both scenarios"""
    
    test_runner = AllRecipesApplePieTest()
    results = await test_runner.run_both_tests(
        test_runner.first_task_description, 
        test_runner.second_task_description
    )
    
    # Initialize logger and log results to JSON
    logger = TestLogger("allrecipes_apple_pie")
    
    # Still generate visualizations but don't print detailed results
    for test_name, test_result in [("First Test", results["first_test"]), 
                                    ("Second Test", results["second_test"])]:
        # Handle screenshot visualization if coordinates are available
        if test_result['agent_result'].get('last_coordinates'):
            x, y = test_result['agent_result']['last_coordinates']
            
            # Use screenshot_path from agent result if available
            screenshot_path = test_result['agent_result'].get('screenshot_path')
            if screenshot_path:
                test_runner.visualize_prediction(
                    image_path=screenshot_path,
                    pred_x=x,
                    pred_y=y,
                    run_id=None,
                    gt_x1=0,
                    gt_y1=0,
                    gt_x2=0,
                    gt_y2=0,
                    is_in_bbox=test_result['evaluation']['passed']
                )
    
    # Log results to JSON file and print summary
    logger.log_and_print_summary(results)
    
    # Return results instead of using assert
    return results

async def main():
    """Main function to run the tests"""
    # Run the test and return results
    results = await test_allrecipes_apple_pie()
    return results

if __name__ == "__main__":
    results = asyncio.run(main())