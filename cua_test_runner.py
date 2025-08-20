from agent import ComputerAgent
from computer import Computer
import os

async with Computer(
    os_type="linux",
    provider_type="cloud",
    name="your-container-name",
    api_key="your-api-key"
) as computer:
    # Take screenshot
    screenshot = await computer.interface.screenshot()

    # Click and type
    await computer.interface.left_click(100, 100)
    await computer.interface.type("Hello!")

agent = ComputerAgent(
    model="anthropic/claude-3-5-sonnet-20241022",
    tools=[computer],
    max_trajectory_budget=5.0
)

messages = [{"role": "user", "content": "Take a screenshot and tell me what you see"}]

async for result in agent.run(messages):
    for item in result["output"]:
        if item["type"] == "message":
            print(item["content"][0]["text"])