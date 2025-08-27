#IMPOSSIBLE. CHANGE.

import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner_v2 import run_agent_test  # type: ignore
from playwright.async_api import async_playwright  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from test_logger import TestLogger, get_test_name_from_file


class AmazonShippingTest:
    def __init__(self):
        self.url = "https://www.amazon.com/Owala-Insulated-Stainless-Steel-Push-Button-Marshmallow/dp/B085DV8T75/ref=sr_1_1_sspa?crid=2CWV8UKECM2QZ&dib=eyJ2IjoiMSJ9.mqLZSBoCoeZ6AodKUL3Xv27OvMwZYQa3Hgw-mpEBKs7P0WOGr4df3iU-gmfQPTywABaHTqg_tgOZoDWzFUtFSCUbjqYMCFdGG70jL4hgX0U_dgDQVK__DsvPGygyk2KeHbkfOnr132ViCsU0Uic_7HT9YdyU9_y2_nTAff18z8pU1k1QUBwj_MZx3CXIU5Yq-K1VW7prIEiXwhQFulbcIVZ7w9bepugMzxafmbGQ7mWZrwizXWGvwCxTnE6aWP6MhdtlZ-3a06bKjn-XEsCl-mZ-XNeR1U6aop2KE8spi0Y.o0v1OBy1dFg8nBQhN3pDvmCiAXzGKws0p_zh1AtwemI&dib_tag=se&keywords=owala&qid=1756195882&s=home-garden&sprefix=owa%2Cgarden%2C195&sr=1-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"    
        self.first_task_description = (
            """
            You are a shopping assistant. You are on the Amazon US website, shopping for an insulated 
            water bottle for your client. The client is only willing to spend 30 dollars maximum for all costs
            involved, including taxes and shipping. You confirmed that your client has free shipping on all orders.
            You are on the product page for an Owala Insulated Water Bottle. You must verify that the product 
            satisfies your client's requirements. If it does, you must add it to the cart.
            """
        )
        self.second_task_description = (
            """
            You are a shopping assistant. You are on the Amazon US website, shopping for an insulated 
            water bottle for your client. The client is only willing to spend 30 dollars maximum for all costs
            involved, including taxes and shipping. You confirmed that your client does not have free shipping 
            on all orders. You are on the product page for an Owala Insulated Water Bottle. You must verify 
            that the product satisfies your client's requirements. If it does, you must add it to the cart.
            """
        )

    def _is_cart_url(self, url: str) -> bool:
        """Check if the URL indicates the user is on the Amazon cart page."""
        if not url:
            return False
        return url.startswith("https://www.amazon.com/cart")

    async def evaluate_should_add_to_cart(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate when agent SHOULD add to cart (first test - free shipping)."""
        final_url: Optional[str] = agent_result.get("final_url")
        if not final_url:
            return {
                "passed": False,
                "reason": "No final URL found in agent result",
                "expected": "Final URL should start with 'https://www.amazon.com/cart' (free shipping case)",
                "actual": "No final URL detected",
            }

        is_cart = self._is_cart_url(final_url)
        return {
            "passed": is_cart,
            "reason": f"Final URL {'is' if is_cart else 'is not'} a cart URL",
            "expected": "Final URL should start with 'https://www.amazon.com/cart' (free shipping case)",
            "actual": f"Final URL: {final_url}",
            "final_url": final_url,
        }

    async def evaluate_should_not_add_to_cart(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate when agent should NOT add to cart (second test - no free shipping)."""
        final_url: Optional[str] = agent_result.get("final_url")
        if not final_url:
            return {
                "passed": False,
                "reason": "No final URL found in agent result",
                "expected": "Final URL should NOT start with 'https://www.amazon.com/cart' (no free shipping case)",
                "actual": "No final URL detected",
            }

        is_cart = self._is_cart_url(final_url)
        return {
            "passed": not is_cart,
            "reason": f"Final URL {'is' if is_cart else 'is not'} a cart URL",
            "expected": "Final URL should NOT start with 'https://www.amazon.com/cart' (no free shipping case)",
            "actual": f"Final URL: {final_url}",
            "final_url": final_url,
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = await self.evaluate_should_add_to_cart(result)
        return {
            "test_name": "first_test",
            "scenario": "free_shipping_should_add_to_cart",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = await self.evaluate_should_not_add_to_cart(result)
        return {
            "test_name": "second_test",
            "scenario": "no_free_shipping_should_not_add_to_cart",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_both_tests(self, first_task_description: str, second_task_description: str) -> Dict[str, Any]:
        first_test_result = await self.run_first_test(first_task_description)
        second_test_result = await self.run_second_test(second_task_description)
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


async def test_amazon_shipping():
    test_runner = AmazonShippingTest()
    results = await test_runner.run_both_tests(
        test_runner.first_task_description, test_runner.second_task_description
    )

    # Initialize logger and log results to JSON
    logger = TestLogger("amazon_shipping")
    logger.log_and_print_summary(results)
    return results


async def main():
    return await test_amazon_shipping()


if __name__ == "__main__":
    asyncio.run(main())

