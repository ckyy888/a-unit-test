import asyncio
import base64
import requests
import os
import re
from playwright.async_api import async_playwright
import time

from ui_tars.action_parser import parse_action_to_structure_output
from ui_tars.prompt import COMPUTER_USE_DOUBAO

## ------------------- CONFIGURATION ------------------- ##

# The base URL of your SGLang server from Cloudflare Tunnel
SGLANG_BASE_URL = "https://bring-determines-france-cambridge.trycloudflare.com"

# The multi-modal model name you are serving
MODEL_NAME = "ui-tars-1.5"

# The specific MiniWoB++ task we will solve
# You can find more tasks at https://miniwob.far.ai/
TASK_URL = "https://miniwob.farama.org/demos/miniwob/circle-center.html"

class UITarsActions:
    """A class to hold the Playwright actions for the MiniWoB++ task."""

    async def extract_actions_and_perform(self, page, response):
        """
        Extracts the Playwright actions from the model response.
        This is a placeholder method; actual implementation will depend on your model's output format.
        """
        # Assuming the response is a string of Playwright code

        if 'click' in response:
            # Extract content between single quotes click(start_box='(86,105)') --> [86,105]
            click_match = response.split("\'(")[-1].split(')')[0]
            if click_match:
                x, y = map(int, click_match.split(','))
                await self.click_xy(page, x, y)

        return response.strip()
    
    async def click(self, page, selector):
        await page.click(selector)

    async def click_xy(self, page, x, y):
        await page.mouse.click(x, y)

    async def type(self, page, selector, text):
        await page.fill(selector, text)

    async def select(self, page, selector, value):
        await page.select_option(selector, value)

    async def wait_for_selector(self, page, selector):
        await page.wait_for_selector(selector)


## ------------------- API FUNCTIONS ------------------- ##

def encode_image_from_bytes(image_bytes: bytes) -> str:
    """Encodes image bytes to a Base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def ask_vision_model_for_action(image_bytes: bytes, task_prompt: str) -> str:
    """
    Sends the screenshot and prompt to the SGLang server and returns the
    generated Playwright code as a string.
    """
    base64_image = encode_image_from_bytes(image_bytes)
    
    api_url = f"{SGLANG_BASE_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "temperature": 0.0, # Zero temperature for deterministic code output
        "max_tokens": 150
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        generated_code = result['choices'][0]['message']['content']
        
        # Clean up the code in case the model wraps it in markdown
        cleaned_code = generated_code.strip().replace('```python', '').replace('```', '').strip()
        return cleaned_code
        
    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred calling the API: {e}")
        return ""


## ------------------- MAIN AGENT SCRIPT ------------------- ##

async def run_miniwob_agent():
    """Main async function to run the Playwright agent."""
    async with async_playwright() as p:
        # Launch the browser in non-headless mode so we can watch
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        page = await browser.new_page()
        interact_tool = UITarsActions()

        # 1. Navigate to the task
        print(f" navigatiing to MiniWoB++ task: {TASK_URL}")
        await page.goto(TASK_URL)
        
        # Wait for the task description to be visible


        
        # 3. Think: Prepare the prompt and ask the model for the next action
        prompt = COMPUTER_USE_DOUBAO + """
        You are an expert at solving MiniWoB++ tasks using Playwright.
        Your task is to interact with the webpage to complete the MiniWoB++ task.
        Press START to start the game. Find and click on the center of the circle, your click will appear as a blue dot. Once the blue
        dot is placed, press submit if you are happy with the placement, or click again to adjust."""
        while True:
            # 2. Perceive: Take a screenshot of the current state
            screenshot_bytes = await page.screenshot()
            generated_code = ask_vision_model_for_action(screenshot_bytes, prompt)
            
            if not generated_code:
                print("❌ Could not get a valid action from the model. Exiting.")
                await browser.close()
                return
            
            print(f"💬 Generated code: {generated_code}")

            # 4. Act: Execute the code generated by the model
            try:
                await interact_tool.extract_actions_and_perform(page=page, response=generated_code)
            except Exception as e:
                print(f"❌ Failed to execute the generated code: {e}")

            time.sleep(1)

        # 5. Verify: Pause and take a final screenshot
        print("✅ Task should be complete. Taking final screenshot to 'result.png'...")
        await page.wait_for_timeout(3000) # Pause for 3 seconds to observe
        await page.screenshot(path="result.png")
        
        await browser.close()
        print("✅ Browser closed. Script finished.")

if __name__ == "__main__":
    asyncio.run(run_miniwob_agent())