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
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from test_logger import TestLogger, get_test_name_from_file

class CourseEraCareersTest:
    def __init__(self):
        self.first_url = "https://www.coursera.org/career-academy?trk_ref=globalnav&category=business&level=advanced"
        self.second_url = "https://www.coursera.org/career-academy?trk_ref=globalnav&category=software_and_it&level=advanced"
        self.first_task_description = """
        You are a personal assistant, whose user has asked you to find a course for them on Coursera. They would like to take an advanced course
        in business. You have navigated to the Coursera career academy page. You just filtered the courses by "Advanced" and "Business" to
        find a course for them that satisfies their request. Once you find a course, you must click on the course to view the course details.
        """
        self.second_task_description = """
        You are a personal assistant, whose user has asked you to find a course for them on Coursera. They would like to take an advanced course
        in software engineering and IT. They would like to take a course that has the highest market demand. You have navigated to the Coursera career academy 
        page. You just filtered the courses by "Advanced" and "Software Engineering & IT" to find a course for them that 
        satisfies their request of finding a course with the highest market demand. Once you find a course, you must click on the course to 
        view the course details.
        """

    def evaluate_course_selection(self, agent_result: Dict[str, Any], expected_url: str, test_name: str) -> Dict[str, Any]:
        """Evaluate if the agent correctly navigated to the expected Coursera course URL"""
        
        # Get final URL from agent result
        final_url = agent_result.get('final_url', '')
        print(f"Evaluating {test_name} with final URL: {final_url}")
        
        # Check if final URL matches the expected URL
        url_matches_expected = final_url == expected_url
        
        return {
            "passed": url_matches_expected,
            "reason": f"URL {'matches' if url_matches_expected else 'does not match'} expected URL - agent {'successfully' if url_matches_expected else 'failed to'} navigate to correct course",
            "expected": f"Expected URL: {expected_url}",
            "actual": f"Final URL: {final_url}",
            "final_url": final_url,
            "url_matches_expected": url_matches_expected
        }
  
    async def run_first_test(self) -> Dict[str, Any]:
        """Run first test - agent should navigate to logistics/supply chain analyst course"""
        print("Running first test (should navigate to logistics/supply chain analyst course)...")
        
        # Expected URL for the first test
        expected_url = "https://www.coursera.org/career-academy/roles/logistics-supply-chain-analyst?level=advanced&recommenderId=role-ranker"
        
        # Run the agent with the first URL and task description
        result = await run_agent_test(self.first_url, self.first_task_description)
        
        # Evaluate the result
        evaluation = self.evaluate_course_selection(result, expected_url, "first_test")
        
        return {
            "test_name": "first_test",
            "scenario": "logistics_supply_chain_analyst",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
        
    async def run_second_test(self) -> Dict[str, Any]:
        """Run second test - agent should navigate to automation engineer course"""
        print("Running second test (should navigate to automation engineer course)...")
        
        # Expected URL for the second test
        expected_url = "https://www.coursera.org/career-academy/roles/automation-engineer?level=advanced&recommenderId=role-ranker"
        
        # Run the agent with the second URL and task description
        result = await run_agent_test(self.second_url, self.second_task_description)
        
        # Evaluate the result
        evaluation = self.evaluate_course_selection(result, expected_url, "second_test")
        
        return {
            "test_name": "second_test", 
            "scenario": "automation_engineer",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
    
    async def run_both_tests(self) -> Dict[str, Any]:
        """Run both test scenarios"""
        print("Starting Coursera Careers Test Suite...")
        
        # Run both tests
        first_test_result = await self.run_first_test()
        second_test_result = await self.run_second_test()
        
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

async def test_coursera_careers():
    """Main test function that runs both scenarios"""
    
    test_runner = CourseEraCareersTest()
    results = await test_runner.run_both_tests()
    
    # Initialize logger and log results to JSON
    logger = TestLogger("coursera_careers")
    
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
    results = await test_coursera_careers()
    return results

if __name__ == "__main__":
    results = asyncio.run(main())