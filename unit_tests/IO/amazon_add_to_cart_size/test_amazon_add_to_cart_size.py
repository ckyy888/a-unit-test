import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


ADD_TO_CART_BIG = {"x": 2400, "y": 650, "width": 200, "height": 60}
ADD_TO_CART_SMALL = {"x": 2400, "y": 730, "width": 100, "height": 40}


class AmazonAddToCartSizeTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.big_path = os.path.join(self.screenshots_dir, 'big.png')
        self.small_path = os.path.join(self.screenshots_dir, 'small.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (2940, 1596), color='white')

    def get_big(self):
        return self._get(self.big_path)

    def get_small(self):
        return self._get(self.small_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]], bbox: Dict[str, float]) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No click detected", "expected": "Click Add to Cart", "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, bbox)
        return {"passed": ok, "reason": f"Click ({x},{y}) {'inside' if ok else 'outside'} Add to Cart", "expected": "Click Add to Cart", "actual": f"({x},{y})"}

    async def run_big(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_big, task)
        evaluation = self.evaluate(result.get('last_coordinates'), ADD_TO_CART_BIG)
        return {"test_name": "big", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_small(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_small, task)
        evaluation = self.evaluate(result.get('last_coordinates'), ADD_TO_CART_SMALL)
        return {"test_name": "small", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_big(task)
        b = await self.run_small(task)
        return {"overall_passed": a["passed"] and b["passed"], "big": a, "small": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_amazon_add_to_cart_size():
    task = (
        """
        Click the "Add to Cart" button regardless of size (large or small).
        """
    )
    t = AmazonAddToCartSizeTest()
    results = await t.run_both(task)
    print("\nAMAZON ADD TO CART SIZE RESULTS")
    for name in ("big", "small"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_amazon_add_to_cart_size()


if __name__ == "__main__":
    asyncio.run(main())

