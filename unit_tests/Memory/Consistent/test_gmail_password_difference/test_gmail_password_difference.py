import asyncio
import os
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, Any, Optional
import sys
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from cua_test_runner import run_agent_test

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from test_logger import TestLogger, get_test_name_from_file

PASSWORD_INPUT_BBOX = {
    "x": 1519,
    "y": 664,
    "width": 948,
    "height": 112,
}

NEXT_BUTTON_BBOX = {
    "x": 2308,
    "y": 967,
    "width": 159,
    "height": 79,
}

class GmailPasswordDifferenceTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.gmail1_path = os.path.join(self.screenshots_dir, 'gmail1.png')
        self.gmail2_path = os.path.join(self.screenshots_dir, 'gmail2.png')
        self.gmail3_path = os.path.join(self.screenshots_dir, 'gmail3.png')
        self.gmail4_path = os.path.join(self.screenshots_dir, 'gmail4.png')
        
        # State tracking for progressive screenshot feeding
        self.current_screenshot_stage = 1
        self.password_input_clicked = False
        self.typing_count = 0
        self.expected_password = ""
        
    def reset_state(self):
        """Reset test state for a new test run"""
        self.current_screenshot_stage = 1
        self.password_input_clicked = False
        self.typing_count = 0
    
    def get_progressive_screenshot(self):
        """Return the appropriate screenshot based on current state"""
        screenshot_paths = {
            1: self.gmail1_path,
            2: self.gmail2_path,
            3: self.gmail3_path,
            4: self.gmail4_path
        }
        
        screenshot_path = screenshot_paths.get(self.current_screenshot_stage, self.gmail1_path)
        print(f"Returning screenshot stage {self.current_screenshot_stage}: {os.path.basename(screenshot_path)}")
        
        try:
            img = Image.open(screenshot_path)
            return img
        except Exception as e:
            print(f"Error loading screenshot {screenshot_path}: {e}")
            return Image.new('RGB', (1920, 1080), color='white')
    
    def is_point_inside_bbox(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        """Check if a point (x, y) is inside the given bounding box"""
        left = bbox["x"]
        right = bbox["x"] + bbox["width"]
        top = bbox["y"]
        bottom = bbox["y"] + bbox["height"]
        return (left <= x <= right and top <= y <= bottom)
    
    def is_click_in_password_input(self, x: int, y: int) -> bool:
        """Check if click is inside password input box"""
        return self.is_point_inside_bbox(x, y, PASSWORD_INPUT_BBOX)
    
    def is_click_in_next_button(self, x: int, y: int) -> bool:
        """Check if click is inside next button"""
        return self.is_point_inside_bbox(x, y, NEXT_BUTTON_BBOX)
    
    def evaluate_password_test(self, agent_result: Dict[str, Any], expected_password: str) -> Dict[str, Any]:
        """Evaluate if the agent successfully entered the correct password"""
        typed_text = agent_result.get('typed_text', '')
        last_coordinates = agent_result.get('last_coordinates')
        reasoning_steps = agent_result.get('reasoning_steps', [])
        action_steps = agent_result.get('action_steps', [])
        
        # Check if password was typed correctly
        password_correct = typed_text.strip() == expected_password
        
        # Check if final click was on next button (for logging purposes)
        next_button_clicked = False
        if last_coordinates:
            x, y = last_coordinates
            next_button_clicked = self.is_click_in_next_button(x, y)
        
        # Agent passes if they typed correct password (no next button requirement)
        passed = password_correct
        
        return {
            "passed": passed,
            "password_correct": password_correct,
            "next_button_clicked": next_button_clicked,
            "final_screenshot_stage": self.current_screenshot_stage,
            "typed_password": typed_text,
            "expected_password": expected_password,
            "last_click_coordinates": last_coordinates,
            "reasoning_steps": reasoning_steps,
            "action_steps": action_steps,
            "total_reasoning_steps": len(reasoning_steps),
            "total_action_steps": len(action_steps),
            "reason": self._get_failure_reason(password_correct, next_button_clicked, self.current_screenshot_stage)
        }
    
    def _get_failure_reason(self, password_correct: bool, next_button_clicked: bool, final_stage: int) -> str:
        """Get detailed reason for test failure or success"""
        if password_correct and next_button_clicked and final_stage == 4:
            return "Test passed: Correct password typed and next button clicked on final screen"
        
        reasons = []
        if not password_correct:
            reasons.append("incorrect password typed")
        if not next_button_clicked:
            reasons.append("next button not clicked")
        if final_stage != 4:
            reasons.append(f"did not reach final screenshot stage (reached stage {final_stage})")
        
        return f"Test failed: {', '.join(reasons)}"

    async def run_password_test(self, expected_password: str, task_description: str) -> Dict[str, Any]:
        """Run a single password test with progressive screenshot feeding"""
        
        self.reset_state()
        self.expected_password = expected_password
        
        # Create a custom screenshot function that tracks state and returns progressive screenshots
        def progressive_screenshot_func():
            return self.get_progressive_screenshot()
        
        # We need to override the run_agent_test behavior to handle progressive feeding
        # But we'll do it by creating a custom screenshot function that responds to agent actions
        
        # Import what we need from cua_test_runner
        from cua_test_runner import AgentTestTracker, screenshot_wrapper
        from agent import ComputerAgent
        
        # Create tracker instance for this test run
        tracker = AgentTestTracker()
        
        # Create wrapped screenshot function
        wrapped_screenshot = screenshot_wrapper(progressive_screenshot_func)
        
        # Initialize reasoning and action logging
        reasoning_steps = []
        action_steps = []
        
        def click_handler(x, y, button='left'):
            tracker.last_action = 'click'
            
            # Convert normalized coordinates (0-1) to pixel coordinates
            screen_width, screen_height = 2940, 1596
            
            # Handle both normalized and pixel coordinates
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Input is normalized, convert to pixels
                pixel_x = int(x * screen_width)
                pixel_y = int(y * screen_height)
                print(f"Converted normalized ({x:.3f}, {y:.3f}) to pixel ({pixel_x}, {pixel_y})")
            else:
                # Input is in pixels - scale from agent's assumed 1440x800 to actual 2940x1596
                scale_x = 2940 / 1440  # 2.042
                scale_y = 1596 / 800   # 1.995
                pixel_x = int(x * scale_x)
                pixel_y = int(y * scale_y)
                print(f"Scaled coordinates: ({x}, {y}) -> ({pixel_x}, {pixel_y})")
            
            # If click coordinates are (0, 0), it might mean click at current position
            if pixel_x == 0 and pixel_y == 0 and tracker.last_coordinates is not None:
                print(f"Click at (0, 0) detected - using current mouse position")
                print(f"Current mouse position from last move: {tracker.last_coordinates}")
                # Keep the move coordinates as the effective click location
                effective_coords = tracker.last_coordinates
            else:
                # Update to the explicit click coordinates
                tracker.last_coordinates = (pixel_x, pixel_y)
                effective_coords = (pixel_x, pixel_y)
            
            print(f"Final click coordinates: {effective_coords}")
            
            # Check if click is in password input box
            if self.is_click_in_password_input(effective_coords[0], effective_coords[1]):
                print("Click detected in password input box!")
                self.password_input_clicked = True
                # Stay on gmail1.png when password input is clicked - don't advance stage
            elif self.is_click_in_next_button(effective_coords[0], effective_coords[1]):
                print("Click detected on next button!")
                # Terminate after any next button click
                tracker.should_terminate = True
                print("Setting termination flag - next button clicked")
            else:
                # For other clicks, set termination flag after any click (default behavior)
                tracker.should_terminate = True
                print("Setting termination flag after click")
            
            return None
        
        def move_handler(x, y):
            # Convert normalized coordinates (0-1) to pixel coordinates
            screen_width, screen_height = 2940, 1596
            
            # Handle both normalized and pixel coordinates
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Input is normalized, convert to pixels
                pixel_x = int(x * screen_width)
                pixel_y = int(y * screen_height)
                print(f"Move: Converted normalized ({x:.3f}, {y:.3f}) to pixel ({pixel_x}, {pixel_y})")
            else:
                # Input is in pixels - scale from agent's assumed 1440x800 to actual 2940x1596
                scale_x = 2940 / 1440  # 2.042
                scale_y = 1596 / 800   # 1.995
                pixel_x = int(x * scale_x)
                pixel_y = int(y * scale_y)
                print(f"Move: Scaled coordinates: ({x}, {y}) -> ({pixel_x}, {pixel_y})")
            
            tracker.last_coordinates = (pixel_x, pixel_y)
            return None
        
        def type_handler(text):
            tracker.last_action = 'type'
            tracker.typed_text += text
            tracker.typing_count += len(text)
            print(f"Typed: '{text}' (total typed: '{tracker.typed_text}', count: {tracker.typing_count})")
            
            # Advance screenshot stage based on typing count, but only if password input was clicked
            if self.password_input_clicked:
                if len(tracker.typed_text) == 1 and self.current_screenshot_stage == 1:
                    self.current_screenshot_stage = 2
                    print("Advanced to screenshot stage 2 (gmail2.png)")
                elif len(tracker.typed_text) == 2 and self.current_screenshot_stage == 2:
                    self.current_screenshot_stage = 3
                    print("Advanced to screenshot stage 3 (gmail3.png)")
                elif len(tracker.typed_text) == 3 and self.current_screenshot_stage == 3:
                    self.current_screenshot_stage = 4
                    print("Advanced to screenshot stage 4 (gmail4.png)")
                    
                # Check if password is complete and terminate if correct
                if tracker.typed_text.strip() == expected_password:
                    print(f"Password '{expected_password}' completed correctly - setting termination flag")
                    tracker.should_terminate = True
            
            return None
        
        # Create dict-based computer handler
        custom_computer = {
            'screenshot': wrapped_screenshot,
            'dimensions': (2940, 1596),
            'click': click_handler,
            'move': move_handler,
            'type': type_handler,
            'wait': lambda ms=1000: print(f"Waiting {ms}ms")
        }
        
        # Create agent
        try:
            agent = ComputerAgent(
                model="claude-sonnet-4-20250514",
                tools=[custom_computer]
            )
        except Exception as e:
            print(f"Error creating agent: {e}")
            return {
                'success': False,
                'error': f"Agent creation failed: {str(e)}",
                'last_action': tracker.last_action,
                'last_coordinates': tracker.last_coordinates,
                'typed_text': tracker.typed_text,
                'typing_count': tracker.typing_count
            }
        
        print(f"Starting Gmail password test with expected password: '{expected_password}'")
        print(f"Task description: {task_description}")
        
        # Run agent
        try:
            print("Running agent...")
            messages = [{"role": "user", "content": task_description}]
            
            # Collect all results from the async generator
            final_result = None
            async for result in agent.run(messages):
                final_result = result
                
                # Process and log reasoning/actions from agent output
                if "output" in result:
                    for item in result["output"]:
                        if item.get("type") == "computer_call":
                            action = item.get("action", {})
                            action_type = action.get("type", "")
                            action_steps.append({
                                'step': len(action_steps) + 1,
                                'action_type': action_type,
                                'action_details': action,
                                'timestamp': result.get('timestamp') if 'timestamp' in result else None
                            })
                            print(f"Computer Action Step {len(action_steps)}: {action_type}({action})")
                        elif item.get("type") != "computer_call_output":
                            # This is reasoning text
                            reasoning_text = item
                            reasoning_steps.append({
                                'step': len(reasoning_steps) + 1,
                                'reasoning': reasoning_text,
                                'timestamp': result.get('timestamp') if 'timestamp' in result else None
                            })
                            print(f"Reasoning Step {len(reasoning_steps)}: {reasoning_text}")
                
                # Check if we should terminate after processing this result
                if tracker.should_terminate:
                    print("Terminating trajectory")
                    break
            
            print(f"Agent completed. Last action: {tracker.last_action}, Last coordinates: {tracker.last_coordinates}")
            print(f"Typed text: '{tracker.typed_text}', Final screenshot stage: {self.current_screenshot_stage}")
            print(f"Captured {len(reasoning_steps)} reasoning steps and {len(action_steps)} action steps")
            
            return {
                'success': True,
                'result': final_result,
                'last_action': tracker.last_action,
                'last_coordinates': tracker.last_coordinates,
                'typed_text': tracker.typed_text,
                'typing_count': tracker.typing_count,
                'reasoning_steps': reasoning_steps,
                'action_steps': action_steps
            }
        except Exception as e:
            print(f"Agent failed with error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'last_action': tracker.last_action,
                'last_coordinates': tracker.last_coordinates,
                'typed_text': tracker.typed_text,
                'typing_count': tracker.typing_count,
                'reasoning_steps': reasoning_steps,
                'action_steps': action_steps
            }

async def test_gmail_password_difference():
    """Main test function that runs both password scenarios"""
    
    # Create test instance
    test_runner = GmailPasswordDifferenceTest()
    
    # Test 1: Password "abc"
    print("\n" + "="*60)
    print("GMAIL PASSWORD TEST 1: Password 'abc'")
    print("="*60)
    
    task_description_abc = """
    You are an AI assistant helping to log into a Gmail account.
    The screen resolution is 2940x1596.
    
    TASK: Log in to Gmail with the password "abc"
    
    INSTRUCTIONS:
    1. Take a screenshot to see the current login page
    2. Click on the password input field
    3. Type the password "abc" (one character at a time)
    
    You have access to computer control functions including screenshot, click, type, scroll, and move.
    Start by taking a screenshot to see the current page.
    """
    
    result_abc = await test_runner.run_password_test("abc", task_description_abc)
    evaluation_abc = test_runner.evaluate_password_test(result_abc, "abc")
    
    # Test 2: Password "xyz"
    print("\n" + "="*60)
    print("GMAIL PASSWORD TEST 2: Password 'xyz'")
    print("="*60)
    
    task_description_xyz = """
    You are an AI assistant helping to log into a Gmail account.
    The screen resolution is 2940x1596.
    
    TASK: Log in to Gmail with the password "xyz"
    
    INSTRUCTIONS:
    1. Take a screenshot to see the current login page
    2. Click on the password input field
    3. Type the password "xyz" (one character at a time)
    
    You have access to computer control functions including screenshot, click, type, scroll, and move.
    Start by taking a screenshot to see the current page.
    """
    
    result_xyz = await test_runner.run_password_test("xyz", task_description_xyz)
    evaluation_xyz = test_runner.evaluate_password_test(result_xyz, "xyz")
    
    # Compile results
    results = {
        "overall_passed": evaluation_abc["passed"] and evaluation_xyz["passed"],
        "abc_test": {
            "passed": evaluation_abc["passed"],
            "agent_result": result_abc,
            "evaluation": evaluation_abc
        },
        "xyz_test": {
            "passed": evaluation_xyz["passed"],
            "agent_result": result_xyz,
            "evaluation": evaluation_xyz
        },
        "summary": {
            "abc_test_passed": evaluation_abc["passed"],
            "xyz_test_passed": evaluation_xyz["passed"],
            "total_tests": 2,
            "passed_tests": sum([evaluation_abc["passed"], evaluation_xyz["passed"]])
        }
    }
    
    # Initialize logger and log results to JSON
    logger = TestLogger("gmail_password_difference")
    logger.log_and_print_summary(results)
    
    return results

async def main():
    """Main function to run the tests"""
    results = await test_gmail_password_difference()
    return results

if __name__ == "__main__":
    results = asyncio.run(main())