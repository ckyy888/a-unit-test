import asyncio
from agent import ComputerAgent
from PIL import Image
from typing import Dict, Any, Optional


class AgentTestTracker:
    """Encapsulates test state tracking to avoid global variables"""
    
    def __init__(self):
        self.last_action = None
        self.last_coordinates = None
        self.should_terminate = False
        self.typed_text = ""
        self.typing_count = 0
    
    def reset(self):
        """Reset all tracking state"""
        self.last_action = None
        self.last_coordinates = None
        self.should_terminate = False
        self.typed_text = ""
        self.typing_count = 0

def screenshot_wrapper(screenshot_func):
    """Wrapper to handle both sync and async screenshot functions"""
    call_count = 0
    
    def wrapper():
        nonlocal call_count
        call_count += 1
        print(f"Screenshot wrapper called #{call_count}")
        
        # If it's a sync function, call it directly
        if not asyncio.iscoroutinefunction(screenshot_func):
            result = screenshot_func()
            print(f"Screenshot taken, type: {type(result)}")
            return result
        else:
            # If it's async, we need to handle it differently
            # This shouldn't happen with our fixed version, but just in case
            print("Warning: Async screenshot function detected, trying to handle...")
            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(screenshot_func())
                return result
            except Exception as e:
                print(f"Error handling async screenshot: {e}")
                # Return a dummy image as fallback
                return Image.new('RGB', (1920, 1080), color='white')
    return wrapper

async def run_agent_test(screenshot_func, task_description: str) -> Dict[str, Any]:
    """Run the agent with a custom screenshot function and return the results"""
    
    # Create tracker instance for this test run
    tracker = AgentTestTracker()
    
    # Create wrapped screenshot function
    wrapped_screenshot = screenshot_wrapper(screenshot_func)
    
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
        
        # Set termination flag after any click
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
            model="claude-3-5-sonnet-20241022",
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
    
    print(f"Starting agent with task: {task_description}")
    print(f"Custom computer tools availablea {list(custom_computer.keys())}")
    
    # Run agent
    try:
        print("Running agent...")
        messages = [{"role": "user", "content": task_description}]
        
        # Collect all results from the async generator
        final_result = None
        async for result in agent.run(messages):
            final_result = result
            # Process any computer calls that might trigger our record_click function
            if "output" in result:
                for item in result["output"]:
                    if item.get("type") == "computer_call":
                        action = item.get("action", {})
                        action_type = action.get("type", "")
                        print(f"Computer Action: {action_type}({action})")
            
            # Check if we should terminate after processing this result
            if tracker.should_terminate:
                print("Terminating trajectory after click detection")
                break
        
        print(f"Agent completed. Last action: {tracker.last_action}, Last coordinates: {tracker.last_coordinates}")
        print(f"Typed text: '{tracker.typed_text}', Typing count: {tracker.typing_count}")
        return {
            'success': True,
            'result': final_result,
            'last_action': tracker.last_action,
            'last_coordinates': tracker.last_coordinates,
            'typed_text': tracker.typed_text,
            'typing_count': tracker.typing_count
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
            'typing_count': tracker.typing_count
        }