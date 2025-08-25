import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


LICENSE_BBOX_HEADER = {"x": 400, "y": 300, "width": 500, "height": 120}
LICENSE_BBOX_DETAILS = {"x": 400, "y": 900, "width": 500, "height": 400}  # nested accordions


class HuggingfaceModelCardNavigationTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.header_path = os.path.join(self.screenshots_dir, 'header.png')
        self.details_path = os.path.join(self.screenshots_dir, 'details.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (1920, 1080), color='white')

    def get_header(self):
        return self._get(self.header_path)

    def get_details(self):
        return self._get(self.details_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]], bbox: Dict[str, float]) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No interaction detected", "expected": "Reveal license info region", "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, bbox)
        return {"passed": ok, "reason": f"Interaction at ({x},{y}) {'inside' if ok else 'outside'} license region", "expected": "Interact with license region", "actual": f"({x},{y})"}

    async def run_header(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_header, task)
        evaluation = self.evaluate(result.get('last_coordinates'), LICENSE_BBOX_HEADER)
        return {"test_name": "license_header_visible", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_details(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_details, task)
        evaluation = self.evaluate(result.get('last_coordinates'), LICENSE_BBOX_DETAILS)
        return {"test_name": "license_hidden_in_details", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_header(task)
        b = await self.run_details(task)
        return {"overall_passed": a["passed"] and b["passed"], "header": a, "details": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_huggingface_model_card_navigation():
    task = (
        """
        Find the model's license. If it's visible in the header, click it.
        If hidden, expand the Model Details accordions to reveal it, then click.
        """
    )
    t = HuggingfaceModelCardNavigationTest()
    results = await t.run_both(task)
    print("\nHUGGINGFACE MODEL CARD NAVIGATION RESULTS")
    for name in ("header", "details"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_huggingface_model_card_navigation()


if __name__ == "__main__":
    asyncio.run(main())

