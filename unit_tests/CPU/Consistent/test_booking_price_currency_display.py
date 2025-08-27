import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner_v2 import run_agent_test
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from test_logger import TestLogger, get_test_name_from_file 


class BookingPriceCurrencyDisplayTest:
    def __init__(self):
        self.first_url = "https://www.booking.com/hotel/gb/capsules-by-zedwell.en-gb.html?aid=336408&label=london-4ILnGvsrD%2AaMG1hpvTmXBQS650650528715%3Apl%3Ata%3Ap13060%3Ap2%3Aac%3Aap%3Aneg%3Afi%3Atikwd-12080521%3Alp9031967%3Ali%3Adec%3Adm%3Appccp%3DUmFuZG9tSVYkc2RlIyh9YavywThF4buZQ9k1iY7Jbs4&sid=35eda8675e10faa2560ef0124f7fe829&all_sr_blocks=1355922903_407898173_0_0_0&checkin=2026-01-01&checkout=2026-01-03&dest_id=-2601889&dest_type=city&dist=0&group_adults=1&group_children=0&hapos=1&highlighted_blocks=1355922903_407898173_0_0_0&hpos=1&matching_block_id=1355922903_407898173_0_0_0&no_rooms=1&req_adults=1&req_children=0&room1=A&sb_price_type=total&sr_order=popularity&sr_pri_blocks=1355922903_407898173_0_0_0__7425&srepoch=1756157921&srpvid=4f70982c0d8705cf&type=total&ucfs=1&"
        self.second_url = "https://www.booking.com/hotel/gb/capsules-by-zedwell.en-gb.html?aid=336408&label=london-4ILnGvsrD*aMG1hpvTmXBQS650650528715%3Apl%3Ata%3Ap13060%3Ap2%3Aac%3Aap%3Aneg%3Afi%3Atikwd-12080521%3Alp9031967%3Ali%3Adec%3Adm%3Appccp%3DUmFuZG9tSVYkc2RlIyh9YavywThF4buZQ9k1iY7Jbs4&sid=35eda8675e10faa2560ef0124f7fe829&all_sr_blocks=1355922903_407898173_0_0_0&checkin=2026-01-01&checkout=2026-01-03&dest_id=-2601889&dest_type=city&dist=0&group_adults=1&group_children=0&hapos=1&highlighted_blocks=1355922903_407898173_0_0_0&hpos=1&matching_block_id=1355922903_407898173_0_0_0&no_rooms=1&req_adults=1&req_children=0&room1=A&sb_price_type=total&sr_order=popularity&sr_pri_blocks=1355922903_407898173_0_0_0__7425&srepoch=1756157921&srpvid=4f70982c0d8705cf&type=total&ucfs=1&selected_currency=JPY"

        self.task_description = (
            """
            You are a travel assistant. You are tasked with booking a room in the Zedwell Capsule Hotel Piccadilly Circus in London 
            for your very particular American client, who will be staying for 2 nights from 2026-01-01 to 2026-01-03. 
            Your client is willing to pay between 100 and 200 dollars in total.
            You have found the hotel's booking page. You are on the hotel page and you have checked everything is correct.
            The only thing left to do is to check the price. After that you are planning to reserve the room immediately.
            """
        )

    @staticmethod
    def evaluate_booking_completion(
        agent_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate by checking if the agent clicked the 'I'll reserve' button."""
        # Check if the agent's reasoning steps mention clicking "I'll reserve"
        reasoning_steps = agent_result.get("reasoning_steps", [])
        clicked_reserve_button = False
        
        for step in reasoning_steps:
            reasoning = step.get("reasoning", {})
            if isinstance(reasoning, dict):
                content = reasoning.get("content", [])
                for content_item in content:
                    if isinstance(content_item, dict):
                        text = content_item.get("text", "")
                        if "I'll reserve" in text or "reserve" in text.lower():
                            clicked_reserve_button = True
                            break
            if clicked_reserve_button:
                break
        
        # Also check if the last action was a click (additional validation)
        last_action = agent_result.get("last_action", "")
        is_click_action = last_action == "click"
        
        # Pass if agent mentioned reserving and performed a click action
        passed = clicked_reserve_button and is_click_action
        
        return {
            "passed": passed,
            "reason": (
                "Successfully clicked 'I'll reserve' button" if passed 
                else f"Did not click 'I'll reserve' button. Last action: {last_action}, Found reserve mention: {clicked_reserve_button}"
            ),
            "expected": "Agent should click the 'I'll reserve' button",
            "actual": f"Last action: {last_action}, Reserve button mentioned: {clicked_reserve_button}",
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.first_url, task_description)
        evaluation = self.evaluate_booking_completion(result)
        return {
            "test_name": "first_test",
            "scenario": "usd_display",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.second_url, task_description)
        evaluation = self.evaluate_booking_completion(result)
        return {
            "test_name": "second_test",
            "scenario": "hkd_display",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_both_tests(self, task_description: str) -> Dict[str, Any]:
        first_test_result = await self.run_first_test(task_description)
        second_test_result = await self.run_second_test(task_description)
        overall_passed = first_test_result["passed"] and second_test_result["passed"]
        return {
            "overall_passed": overall_passed,
            "first_test": first_test_result,
            "second_test": second_test_result,
            "summary": {
                "first_test_passed": first_test_result["passed"],
                "second_test_passed": second_test_result["passed"],
                "total_tests": 2,
                "passed_tests": sum([
                    first_test_result["passed"],
                    second_test_result["passed"],
                ]),
            },
        }


async def test_booking_price_currency_display():
    test_runner = BookingPriceCurrencyDisplayTest()
    results = await test_runner.run_both_tests(
        test_runner.task_description
    )

    # Initialize logger and log results to JSON
    logger = TestLogger("booking_price_currency_display")
    logger.log_and_print_summary(results)
    return results


async def main():
    return await test_booking_price_currency_display()


if __name__ == "__main__":
    asyncio.run(main())

