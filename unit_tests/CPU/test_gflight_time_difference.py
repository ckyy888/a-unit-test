
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

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cua_test_runner_v2 import run_agent_test

class GoogleFlightsTimeDifferenceTest:
    def __init__(self):
        self.url = "https://www.google.com/travel/flights/booking?tfs=CBwQAhpJEgoyMDI1LTA5LTAxIh8KA1NGTxIKMjAyNS0wOS0wMRoDSEtHKgJVQTIDODc3agwIAhIIL20vMGQ2bHByDAgDEggvbS8wM2g2NBpJEgoyMDI1LTA5LTE2Ih8KA0hLRxIKMjAyNS0wOS0xNhoDU0ZPKgJVQTIDODYyagwIAxIIL20vMDNoNjRyDAgCEggvbS8wZDZscEABSAFwAYIBCwj___________8BmAEB&tfu=CmxDalJJYm5KclZGcEdjMDh6V2xsQlMyMXJUR2RDUnkwdExTMHRMUzB0TFhsc2FHVXhPVUZCUVVGQlIybHpUSGxOU25ablRFMUJFZ1ZWUVRnMk1ob0xDSm5wQkJBQ0dnTlZVMFE0SEhDWjZRUT0SAggAIgYKATASATA"
        self.first_task_description = """
        You are a travel agent. You must book a round trip flight for your client from San Francisco to Hong Kong and back that leaves on September 1st from SFO and departs on September 16th from HKG.
        In previous steps, you searched for round trip flights from San Francisco to Hong Kong and back on Google Flights. You found a United Airline flight that seems to be the best option.
        You are now on the flight details page. You must verify that the flight satisfies the user's request. If it does, you must select the flight by clicking the "Select flight" button.
        The flight times look correct - this flight should meet your client's requirements. Please click the "Select flight" button to proceed with booking.
        """
        self.second_task_description = """
        You are a travel agent. You must book a round trip flight for your client from San Francisco to Hong Kong and back that leaves on September 1st from SFO and arrives on September 16th from HKG.
        In previous steps, you searched for round trip flights from San Francisco to Hong Kong and back on Google Flights. You found a United Airline flight that seems to be the best option.
        You are now on the flight details page. You must verify that the flight satisfies the user's request. If it does, you must select the flight by clicking the "Select flight" button.
        The flight times look correct - this flight should meet your client's requirements. Please click the "Select flight" button to proceed with booking.
        """



    def evaluate_flight_selection(self, coordinates: Optional[Tuple[int, int]], agent_result: Dict[str, Any], should_select: bool) -> Dict[str, Any]:
        """Evaluate if the agent correctly selected or didn't select the flight based on final URL"""
        success_urls = ["https://www.united.com/en/us/traveler/choose-travelers?src=uatravel&cartId=1E701CF5-60C5-4FE4-9493-46F8C5882969&type=ext",
                        "https://www.united.com/en/us/traveler/choose-travelers?src=uatravel&cartId=FC8AD58F-8813-4881-88DF-B70515B7D58B&type=ext",
                        "https://www.united.com/en/us/traveler/choose-travelers?src=uatravel&cartId=A27432E5-DAB3-40E7-864D-C47521770A47&type=ext",
                        "https://www.united.com/en/us/traveler/choose-travelers?src=uatravel&cartId=8FFB1AE7-11FC-410D-9628-8583FECCADF7&type=ext",
                        "https://www.united.com/en/us/traveler/choose-travelers?src=uatravel&cartId=153F0A14-4EEF-45DA-BED6-1B3637571E7E&type=ext"]
        
        # Get final URL from agent result
        final_url = agent_result.get('final_url', '')
        print(f"Evaluating with final URL: {final_url}")
        
        # Check if final URL matches any of the success URLs
        url_matches_success = any(final_url.startswith(success_url.split('?')[0]) and 
                                 'choose-travelers' in final_url 
                                 for success_url in success_urls)
        
        if should_select:
            # First test: should succeed if URL matches success URLs
            return {
                "passed": url_matches_success,
                "reason": f"URL {'matches' if url_matches_success else 'does not match'} success pattern - flight {'was' if url_matches_success else 'was not'} selected correctly",
                "expected": "Flight selection (URL should contain 'choose-travelers')",
                "actual": f"Final URL: {final_url}",
                "final_url": final_url,
                "url_matches_success": url_matches_success
            }
        else:
            # Second test: should succeed if URL does NOT match success URLs
            return {
                "passed": not url_matches_success,
                "reason": f"URL {'matches' if url_matches_success else 'does not match'} success pattern - flight {'was incorrectly' if url_matches_success else 'was correctly not'} selected",
                "expected": "No flight selection (URL should NOT contain 'choose-travelers')",
                "actual": f"Final URL: {final_url}",
                "final_url": final_url,
                "url_matches_success": url_matches_success
            }
  
    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        """Run first test - agent should select the flight"""
        print("Running first test (should select flight)...")
        
        # Run the agent with the URL
        result = await run_agent_test(self.url, task_description)
        
        # Evaluate the result - expect flight selection
        evaluation = self.evaluate_flight_selection(None, result, should_select=True)
        
        return {
            "test_name": "first_test",
            "scenario": "should_select_flight",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
        
    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        """Run second test - agent should not select the flight"""
        print("Running second test (should not select flight)...")
        
        # Run the agent with the URL
        result = await run_agent_test(self.url, task_description)
        
        # Evaluate the result - expect no flight selection
        evaluation = self.evaluate_flight_selection(None, result, should_select=False)
        
        return {
            "test_name": "second_test", 
            "scenario": "should_not_select_flight",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"]
        }
    
    async def run_both_tests(self, first_task_description: str, second_task_description: str) -> Dict[str, Any]:
        """Run both test scenarios"""
        print("Starting Google Flights Time Difference Test Suite...")
        
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

async def test_gflight_time_difference():
    """Main test function that runs both scenarios"""
    
    test_runner = GoogleFlightsTimeDifferenceTest()
    results = await test_runner.run_both_tests(
        test_runner.first_task_description, 
        test_runner.second_task_description
    )
    
    # Print detailed results
    print("\n" + "="*50)
    print("GOOGLE FLIGHTS TIME DIFFERENCE TEST RESULTS")
    print("="*50)
    
    for test_name, test_result in [("First Test", results["first_test"]), 
                                    ("Second Test", results["second_test"])]:
        print(f"\n{test_name}:")
        print(f"  Status: {'PASSED' if test_result['passed'] else 'FAILED'}")
        print(f"  Reason: {test_result['evaluation']['reason']}")
        
        # Display final URL information
        final_url = test_result['evaluation'].get('final_url', 'Unknown')
        print(f"  Final URL: {final_url}")
        print(f"  URL matches success pattern: {test_result['evaluation'].get('url_matches_success', False)}")
        
        # Handle screenshot visualization if coordinates are available
        if test_result['agent_result'].get('last_coordinates'):
            x, y = test_result['agent_result']['last_coordinates']
            print(f"  Click coordinates: ({x}, {y})")
            
            # Use screenshot_path from agent result if available
            screenshot_path = test_result['agent_result'].get('screenshot_path')
            if screenshot_path:
                print(f"  Screenshot saved at: {screenshot_path}")
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
            else:
                print("  No screenshot available for visualization")
        else:
            print("  No click detected")
    
    print(f"\nOverall Result: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    print(f"Tests passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
    
    # Return results instead of using assert
    return results

async def main():
    """Main function to run the tests"""
    # Run the test and return results
    results = await test_gflight_time_difference()
    return results

if __name__ == "__main__":
    results = asyncio.run(main())