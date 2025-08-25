import asyncio
import base64
import os
import shutil
import tempfile
from io import BytesIO
from agent import ComputerAgent
from agent.computers import AsyncComputerHandler
from PIL import Image
from typing import Dict, Any, Optional, Literal, List, Union
from playwright.async_api import async_playwright

class PlaywrightComputerHandler(AsyncComputerHandler):
    """Custom computer handler using Playwright for browser automation."""
    
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.browser = None
        self.context = None
        self.page = None
        self.viewport_width = 2940
        self.viewport_height = 1596
        # Target coordinate system dimensions (what coordinates are expected to be in)
        self.target_width = 1440
        self.target_height = 800
        # Scaling factors
        self.scale_x = self.viewport_width / self.target_width  # 2940/1440 = 2.0417
        self.scale_y = self.viewport_height / self.target_height  # 1596/800 = 1.995
        self.screenshot_count = 0
        # Track state internally
        self.last_action = None
        self.last_coordinates = None
        self.should_terminate = False
        # Track screenshots
        self.last_screenshot_path = None
        self.temp_screenshots = []  # Keep track of temp screenshots to clean up
    
    async def _ensure_browser_ready(self):
        """Ensure browser and page are ready for use."""
        if self.browser is None or self.context is None or self.page is None:
            playwright = await async_playwright().start()
            
            # Launch browser with better configuration
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            # Create context with better settings
            self.context = await self.browser.new_context(
                viewport={'width': self.viewport_width, 'height': self.viewport_height},
                ignore_https_errors=True,
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            self.page = await self.context.new_page()
            
            # Navigate with better error handling
            try:
                print(f"Navigating to: {self.url}")
                await self.page.goto(self.url, wait_until='domcontentloaded', timeout=30000)
                
                # Wait for page to be ready
                await self.page.wait_for_load_state('networkidle', timeout=30000)
                print(f"Browser initialized successfully for URL: {self.url}")
                print(f"Current page URL: {self.page.url}")
                
            except Exception as e:
                print(f"Error during page navigation: {e}")
                # Try to get current URL even on error
                try:
                    current_url = self.page.url
                    print(f"Page URL after error: {current_url}")
                except:
                    print("Could not get page URL after error")
                raise
    
    def _scale_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates from target coordinate system to viewport coordinate system."""
        scaled_x = int(x * self.scale_x)
        scaled_y = int(y * self.scale_y)
        print(f"Scaling coordinates: ({x}, {y}) -> ({scaled_x}, {scaled_y})")
        return scaled_x, scaled_y
    
    async def get_environment(self) -> Literal["windows", "mac", "linux", "browser"]:
        """Get the current environment type."""
        return "browser"
    
    async def get_dimensions(self) -> tuple[int, int]:
        """Get screen dimensions as (width, height)."""
        return (self.viewport_width, self.viewport_height)
    
    async def screenshot(self) -> str:
        """Take a screenshot, save to disk, and return as base64 string."""
        await self._ensure_browser_ready()
        self.screenshot_count += 1
        print(f"Screenshot #{self.screenshot_count} for URL: {self.url}")
        
        try:
            # Check page state before screenshot
            current_url = self.page.url
            print(f"Taking screenshot of page at: {current_url}")
            
            # Check if page is in error state
            if 'chrome-error://' in current_url:
                print(f"⚠️  Page is in error state: {current_url}")
                print("This indicates a navigation or loading failure")
            
            # Take screenshot with error handling
            screenshot_bytes = await self.page.screenshot(
                full_page=False,
                timeout=10000  # 10 second timeout
            )
            
            # Check if screenshot is empty/small (indicating blank page)
            if len(screenshot_bytes) < 1000:  # Less than 1KB suggests blank page
                print(f"⚠️  Screenshot is very small ({len(screenshot_bytes)} bytes) - may be blank page")
            
            # Save screenshot to temporary file
            temp_dir = tempfile.gettempdir()
            screenshot_filename = f"screenshot_{self.screenshot_count}_{id(self)}.png"
            screenshot_path = os.path.join(temp_dir, screenshot_filename)
            
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot_bytes)
            
            # Clean up previous screenshot if it exists and wasn't preserved
            if self.last_screenshot_path and os.path.exists(self.last_screenshot_path):
                try:
                    os.remove(self.last_screenshot_path)
                    if self.last_screenshot_path in self.temp_screenshots:
                        self.temp_screenshots.remove(self.last_screenshot_path)
                except OSError:
                    pass  # File might already be deleted
            
            # Update last screenshot path and track it
            self.last_screenshot_path = screenshot_path
            self.temp_screenshots.append(screenshot_path)
            
            # Convert to base64
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            print(f"Screenshot saved to: {screenshot_path}, size: {len(screenshot_bytes)} bytes")
            
            return screenshot_base64
            
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            print(f"Current page URL: {self.page.url}")
            raise
    
    async def click(self, x: int, y: int, button: str = "left") -> None:
        """Click at coordinates with specified button."""
        self.last_action = 'click'
        
        # Handle both normalized and pixel coordinates
        if 0 <= x <= 1 and 0 <= y <= 1:
            # Input is normalized, convert to pixels in target coordinate system first
            target_x = int(x * self.target_width)
            target_y = int(y * self.target_height)
            print(f"Converted normalized ({x:.3f}, {y:.3f}) to target ({target_x}, {target_y})")
            # Then scale to viewport coordinates
            pixel_x, pixel_y = self._scale_coordinates(target_x, target_y)
        else:
            # Input is in target coordinate system, scale to viewport coordinates
            pixel_x, pixel_y = self._scale_coordinates(int(x), int(y))
        
        # If click coordinates are (0, 0), it might mean click at current position
        if pixel_x == 0 and pixel_y == 0 and self.last_coordinates is not None:
            print(f"Click at (0, 0) detected - using current mouse position")
            print(f"Current mouse position from last move: {self.last_coordinates}")
            # Keep the move coordinates as the effective click location
            effective_coords = self.last_coordinates
        else:
            # Update to the explicit click coordinates
            self.last_coordinates = (pixel_x, pixel_y)
            effective_coords = (pixel_x, pixel_y)
        
        print(f"Final click coordinates: {effective_coords}")
        
        # Set termination flag after any click
        self.should_terminate = True
        print("Setting termination flag after click")
        
        # Perform the actual click if browser is ready
        await self._ensure_browser_ready()
        
        # Store current URL to detect navigation
        current_url = self.page.url
        print(f"Current URL before click: {current_url}")
        
        try:
            # Set up detection for new page/tab before clicking
            new_page_promise = self.context.wait_for_event('page', timeout=5000)  # Increased timeout
            
            # Perform the click
            await self.page.click(f'body', position={'x': effective_coords[0], 'y': effective_coords[1]})
            
            try:
                # Wait for new page/tab with timeout
                new_page = await new_page_promise
                print(f"New page/tab detected, waiting for it to load...")
                
                # Wait for new page to load with better error handling
                try:
                    await new_page.wait_for_load_state('domcontentloaded', timeout=15000)
                    await new_page.wait_for_load_state('networkidle', timeout=15000)
                except Exception as load_error:
                    print(f"Warning: New page load timeout: {load_error}")
                    # Continue anyway, page might still be usable
                
                # Switch to the new page
                self.page = new_page
                print(f"New tab/window opened and loaded: {self.page.url}")
                
                # Check if new page is in error state
                if 'chrome-error://' in self.page.url:
                    print(f"⚠️  New page is in error state: {self.page.url}")
                
            except Exception as new_page_error:
                # No new page opened, check if current page navigated
                print(f"No new tab detected: {new_page_error}")
                
                # Wait a bit for potential navigation on current page
                await asyncio.sleep(1)
                
                # Check if navigation occurred on current page
                if self.page.url != current_url:
                    print(f"Navigation detected on current page, waiting for load state...")
                    try:
                        await self.page.wait_for_load_state('domcontentloaded', timeout=15000)
                        await self.page.wait_for_load_state('networkidle', timeout=15000)
                    except Exception as load_error:
                        print(f"Warning: Page load timeout: {load_error}")
                    print(f"Navigation completed to: {self.page.url}")
                else:
                    # Wait a bit more for slow navigation
                    await asyncio.sleep(3)  # Increased wait time
                    if self.page.url != current_url:
                        print(f"Delayed navigation detected, waiting for load state...")
                        try:
                            await self.page.wait_for_load_state('domcontentloaded', timeout=15000)
                            await self.page.wait_for_load_state('networkidle', timeout=15000)
                        except Exception as load_error:
                            print(f"Warning: Delayed page load timeout: {load_error}")
                        print(f"Navigation completed to: {self.page.url}")
                    else:
                        print(f"No navigation detected after click, URL remains: {self.page.url}")
                        
        except Exception as e:
            print(f"Error during click and navigation handling: {e}")
            print(f"Final URL: {self.page.url}")
            # Don't re-raise here, let the test continue
    
    async def double_click(self, x: int, y: int) -> None:
        """Double click at coordinates."""
        await self._ensure_browser_ready()
        # Scale coordinates from target to viewport coordinate system
        scaled_x, scaled_y = self._scale_coordinates(int(x), int(y))
        await self.page.dblclick(f'body', position={'x': scaled_x, 'y': scaled_y})
    
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at coordinates with specified scroll amounts."""
        await self._ensure_browser_ready()
        # Scale coordinates and move to position before scrolling
        scaled_x, scaled_y = self._scale_coordinates(x, y)
        await self.page.mouse.move(scaled_x, scaled_y)
        await self.page.mouse.wheel(scroll_x, scroll_y)
    
    async def type(self, text: str) -> None:
        """Type text."""
        await self._ensure_browser_ready()
        await self.page.keyboard.type(text)
    
    async def wait(self, ms: int = 1000) -> None:
        """Wait for specified milliseconds."""
        print(f"Waiting {ms}ms")
        await asyncio.sleep(ms / 1000)
    
    async def move(self, x: int, y: int) -> None:
        """Move cursor to coordinates."""
        # Handle both normalized and pixel coordinates
        if 0 <= x <= 1 and 0 <= y <= 1:
            # Input is normalized, convert to pixels in target coordinate system first
            target_x = int(x * self.target_width)
            target_y = int(y * self.target_height)
            print(f"Move: Converted normalized ({x:.3f}, {y:.3f}) to target ({target_x}, {target_y})")
            # Then scale to viewport coordinates
            pixel_x, pixel_y = self._scale_coordinates(target_x, target_y)
        else:
            # Input is in target coordinate system, scale to viewport coordinates
            pixel_x, pixel_y = self._scale_coordinates(int(x), int(y))
        
        self.last_coordinates = (pixel_x, pixel_y)
        
        # Perform the actual move if browser is ready
        await self._ensure_browser_ready()
        await self.page.mouse.move(pixel_x, pixel_y)
    
    async def keypress(self, keys: Union[List[str], str]) -> None:
        """Press key combination."""
        await self._ensure_browser_ready()
        if isinstance(keys, str):
            await self.page.keyboard.press(keys)
        else:
            # Handle key combinations
            for key in keys:
                await self.page.keyboard.press(key)
    
    async def drag(self, path: List[Dict[str, int]]) -> None:
        """Drag along specified path."""
        await self._ensure_browser_ready()
        if not path:
            return
        
        # Start drag at first point (scale coordinates)
        start_point = path[0]
        scaled_start_x, scaled_start_y = self._scale_coordinates(start_point['x'], start_point['y'])
        await self.page.mouse.move(scaled_start_x, scaled_start_y)
        await self.page.mouse.down()
        
        # Move through path (scale each point)
        for point in path[1:]:
            scaled_x, scaled_y = self._scale_coordinates(point['x'], point['y'])
            await self.page.mouse.move(scaled_x, scaled_y)
        
        # End drag
        await self.page.mouse.up()
    
    async def get_current_url(self) -> str:
        """Get current URL (for browser environments)."""
        await self._ensure_browser_ready()
        return self.page.url
    
    async def left_mouse_down(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Left mouse down at coordinates."""
        await self._ensure_browser_ready()
        if x is not None and y is not None:
            # Scale coordinates from target to viewport coordinate system
            scaled_x, scaled_y = self._scale_coordinates(x, y)
            await self.page.mouse.move(scaled_x, scaled_y)
        await self.page.mouse.down()
    
    async def left_mouse_up(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Left mouse up at coordinates."""
        await self._ensure_browser_ready()
        if x is not None and y is not None:
            # Scale coordinates from target to viewport coordinate system
            scaled_x, scaled_y = self._scale_coordinates(x, y)
            await self.page.mouse.move(scaled_x, scaled_y)
        await self.page.mouse.up()
    
    def reset_state(self):
        """Reset all tracking state."""
        self.last_action = None
        self.last_coordinates = None
        self.should_terminate = False
    
    def preserve_last_screenshot(self) -> Optional[str]:
        """Preserve the last screenshot by moving it to current directory and return the path."""
        if not self.last_screenshot_path or not os.path.exists(self.last_screenshot_path):
            return None
        
        try:
            # Create visualizations directory in current working directory
            current_dir = os.getcwd()
            screenshots_dir = os.path.join(current_dir, 'screenshots')
            os.makedirs(screenshots_dir, exist_ok=True)
            
            # Create a permanent filename
            filename = f"screenshot_click_{self.screenshot_count}.png"
            permanent_path = os.path.join(screenshots_dir, filename)
            
            # Move the file from temp to permanent location
            shutil.move(self.last_screenshot_path, permanent_path)
            
            # Remove from temp tracking since it's now permanent
            if self.last_screenshot_path in self.temp_screenshots:
                self.temp_screenshots.remove(self.last_screenshot_path)
            
            print(f"Screenshot preserved at: {permanent_path}")
            return permanent_path
            
        except Exception as e:
            print(f"Error preserving screenshot: {e}")
        return None
    
    async def close(self):
        """Clean up browser resources and temporary screenshots."""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.context = None
            self.page = None
        
        # Clean up any remaining temporary screenshots
        for screenshot_path in self.temp_screenshots[:]:  # Use slice copy to avoid modification during iteration
            if os.path.exists(screenshot_path):
                try:
                    os.remove(screenshot_path)
                    print(f"Cleaned up temporary screenshot: {screenshot_path}")
                except OSError:
                    pass  # File might already be deleted
            self.temp_screenshots.remove(screenshot_path)

async def run_agent_test(url: str, task_description: str) -> Dict[str, Any]:
    """Run the agent with a URL to screenshot and return the results"""
    
    # Create class-based computer handler
    custom_computer = PlaywrightComputerHandler(url)
    
    # Create agent
    try:
        agent = ComputerAgent(
            model="claude-3-5-sonnet-20241022",
            tools=[custom_computer]
        )
    except Exception as e:
        print(f"Error creating agent: {e}")
        await custom_computer.close()  # Clean up browser resources
        return {
            'success': False,
            'error': f"Agent creation failed: {str(e)}",
            'last_action': custom_computer.last_action,
            'last_coordinates': custom_computer.last_coordinates
        }
    
    print(f"Starting agent with task: {task_description}")
    print(f"Custom computer handler: {type(custom_computer).__name__}")
    
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
                    if item.get("type") != "computer_call_output":
                        print(f"Everything: {item}")

                    if item.get("type") == "computer_call":
                        action = item.get("action", {})
                        action_type = action.get("type", "")
                        print(f"Computer Action: {action_type}({action})")
                    if item.get("type") == "reasoning":
                        print(f"Reasoning: {item.get('output_text')}")
            
            # Check if we should terminate after processing this result
            if custom_computer.should_terminate:
                print("Terminating trajectory after click detection")
                break
        
        print(f"Agent completed. Last action: {custom_computer.last_action}, Last coordinates: {custom_computer.last_coordinates}")
        
        # Get the final URL before closing browser
        final_url = None
        try:
            final_url = await custom_computer.get_current_url()
            print(f"Final URL: {final_url}")
        except Exception as e:
            print(f"Could not get final URL: {e}")
        
        # Preserve screenshot if there was a click
        screenshot_path = None
        if custom_computer.last_action == 'click' and custom_computer.last_coordinates:
            screenshot_path = custom_computer.preserve_last_screenshot()
        
        # Clean up browser resources
        await custom_computer.close()
        
        result = {
            'success': True,
            'result': final_result,
            'last_action': custom_computer.last_action,
            'last_coordinates': custom_computer.last_coordinates
        }
        
        if final_url:
            result['final_url'] = final_url
        
        if screenshot_path:
            result['screenshot_path'] = screenshot_path
            
        return result
    except Exception as e:
        print(f"Agent failed with error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        # Get the final URL before closing browser, even on error
        final_url = None
        try:
            final_url = await custom_computer.get_current_url()
            print(f"Final URL (on error): {final_url}")
        except Exception as url_e:
            print(f"Could not get final URL on error: {url_e}")
        
        # Preserve screenshot if there was a click, even on error
        screenshot_path = None
        if custom_computer.last_action == 'click' and custom_computer.last_coordinates:
            screenshot_path = custom_computer.preserve_last_screenshot()
        
        # Clean up browser resources
        await custom_computer.close()
        
        result = {
            'success': False,
            'error': str(e),
            'last_action': custom_computer.last_action,
            'last_coordinates': custom_computer.last_coordinates
        }
        
        if final_url:
            result['final_url'] = final_url
        
        if screenshot_path:
            result['screenshot_path'] = screenshot_path
            
        return result