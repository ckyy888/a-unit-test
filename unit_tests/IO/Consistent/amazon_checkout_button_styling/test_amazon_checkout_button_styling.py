import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


BUY_NOW_BBOX_A = {"x": 2400, "y": 720, "width": 420, "height": 80}  # orange rectangular
BUY_NOW_BBOX_B = {"x": 2400, "y": 720, "width": 420, "height": 80}  # green rounded same area


class AmazonCheckoutButtonStylingTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.variant_a_path = os.path.join(self.screenshots_dir, 'variant_a.png')
        self.variant_b_path = os.path.join(self.screenshots_dir, 'variant_b.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (2940, 1596), color='white')

    def get_variant_a(self):
        return self._get(self.variant_a_path)

    def get_variant_b(self):
        return self._get(self.variant_b_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]], bbox: Dict[str, float]) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No click detected", "expected": "Click inside Buy Now", "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, bbox)
        return {"passed": ok, "reason": f"Click at ({x},{y}) {'inside' if ok else 'outside'} Buy Now", "expected": "Click inside Buy Now", "actual": f"({x},{y})"}

    async def run_a(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_variant_a, task)
        evaluation = self.evaluate(result.get('last_coordinates'), BUY_NOW_BBOX_A)
        return {"test_name": "variant_a", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_b(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_variant_b, task)
        evaluation = self.evaluate(result.get('last_coordinates'), BUY_NOW_BBOX_B)
        return {"test_name": "variant_b", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_a(task)
        b = await self.run_b(task)
        return {"overall_passed": a["passed"] and b["passed"], "variant_a": a, "variant_b": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_amazon_checkout_button_styling():
    task = (
        """
        Click the "Buy Now" button to proceed to checkout regardless of color or shape.
        """
    )
    t = AmazonCheckoutButtonStylingTest()
    results = await t.run_both(task)
    print("\nAMAZON CHECKOUT BUTTON STYLING RESULTS")
    for name in ("variant_a", "variant_b"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_amazon_checkout_button_styling()


if __name__ == "__main__":
    asyncio.run(main())

