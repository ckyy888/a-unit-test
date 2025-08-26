#IMPOSSIBLE. CHANGE.

import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cua_test_runner_v2 import run_agent_test  # type: ignore
from playwright.async_api import async_playwright  # type: ignore


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

    async def _get_add_to_cart_bbox(self) -> Optional[Dict[str, float]]:
        """Fetch the absolute bounding box of the add-to-cart button in the same viewport as the agent.

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

                # CSS selector based on provided element
                selector = "#add-to-cart-button"
                await page.wait_for_selector(selector, state="attached", timeout=20000)
                element = await page.query_selector(selector)
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

    async def evaluate_should_add_to_cart(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate when agent SHOULD click add to cart (first test - free shipping)."""
        coordinates: Optional[Tuple[int, int]] = agent_result.get("last_coordinates")
        if not coordinates:
            return {
                "passed": False,
                "reason": "No click action was performed",
                "expected": "Click inside Add to Cart button (free shipping case)",
                "actual": "No click detected",
            }

        bbox = await self._get_add_to_cart_bbox()
        if not bbox:
            return {
                "passed": False,
                "reason": "Could not locate Add to Cart button by selector",
                "expected": "Click inside Add to Cart button (free shipping case)",
                "actual": f"Coordinates: {coordinates}",
            }

        x, y = coordinates
        inside = self._is_inside(x, y, bbox)
        return {
            "passed": inside,
            "reason": f"Click at ({x}, {y}) {'inside' if inside else 'outside'} Add to Cart button",
            "expected": "Click inside Add to Cart button (free shipping case)",
            "actual": f"Click at ({x}, {y})",
            "coordinates": coordinates,
            "add_to_cart_bbox": bbox,
        }

    async def evaluate_should_not_add_to_cart(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate when agent should NOT click add to cart (second test - no free shipping)."""
        coordinates: Optional[Tuple[int, int]] = agent_result.get("last_coordinates")
        
        # If no click was made, that's correct behavior for this test
        if not coordinates:
            return {
                "passed": True,
                "reason": "No click action was performed (correct - shipping costs exceed budget)",
                "expected": "No click on Add to Cart button (no free shipping case)",
                "actual": "No click detected",
            }

        bbox = await self._get_add_to_cart_bbox()
        if not bbox:
            # If we can't find the button but there was a click, assume it wasn't on add-to-cart
            return {
                "passed": True,
                "reason": "Could not locate Add to Cart button, but click was made elsewhere",
                "expected": "No click on Add to Cart button (no free shipping case)",
                "actual": f"Click at coordinates: {coordinates}",
                "coordinates": coordinates,
            }

        x, y = coordinates
        inside = self._is_inside(x, y, bbox)
        return {
            "passed": not inside,
            "reason": f"Click at ({x}, {y}) {'inside' if inside else 'outside'} Add to Cart button",
            "expected": "No click on Add to Cart button (no free shipping case)",
            "actual": f"Click at ({x}, {y})",
            "coordinates": coordinates,
            "add_to_cart_bbox": bbox,
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

    print("\n" + "=" * 50)
    print("AMAZON SHIPPING TEST RESULTS")
    print("=" * 50)
    for test_name, test_result in [
        ("First Test", results["first_test"]),
        ("Second Test", results["second_test"]),
    ]:
        print(f"\n{test_name}:")
        print(f"  Status: {'PASSED' if test_result['passed'] else 'FAILED'}")
        print(f"  Reason: {test_result['evaluation']['reason']}")
        final_url = test_result["evaluation"].get("actual", "Unknown")
        print(f"  Final URL: {final_url}")

    print(
        f"\nOverall Result: {'PASSED' if results['overall_passed'] else 'FAILED'}"
    )
    print(
        f"Tests passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}"
    )
    return results


async def main():
    return await test_amazon_shipping()


if __name__ == "__main__":
    asyncio.run(main())

