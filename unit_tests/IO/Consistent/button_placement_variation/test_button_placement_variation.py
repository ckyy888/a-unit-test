import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


SUBMIT_BBOX_TOP_RIGHT = {"x": 2500, "y": 80, "width": 300, "height": 80}
SUBMIT_BBOX_BOTTOM_CENTER = {"x": 1400, "y": 1450, "width": 300, "height": 80}


class ButtonPlacementVariationTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.variant_a_path = os.path.join(self.screenshots_dir, 'variant_a.png')
        self.variant_b_path = os.path.join(self.screenshots_dir, 'variant_b.png')

    def _get_dummy_or_file(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (2940, 1596), color='white')

    def get_variant_a(self):
        return self._get_dummy_or_file(self.variant_a_path)

    def get_variant_b(self):
        return self._get_dummy_or_file(self.variant_b_path)

    def _is_inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return (bbox["x"] <= x <= bbox["x"] + bbox["width"]) and (
            bbox["y"] <= y <= bbox["y"] + bbox["height"]
        )

    def evaluate_click(self, coordinates: Optional[Tuple[int, int]], bbox: Dict[str, float]) -> Dict[str, Any]:
        if coordinates is None:
            return {
                "passed": False,
                "reason": "No click action was performed",
                "expected": "Click inside Submit button",
                "actual": "No click detected",
            }
        x, y = coordinates
        inside = self._is_inside(x, y, bbox)
        return {
            "passed": inside,
            "reason": f"Click at ({x}, {y}) {'inside' if inside else 'outside'} Submit button",
            "expected": "Click inside Submit button",
            "actual": f"Click at ({x}, {y})",
            "coordinates": coordinates,
        }

    async def run_variant_a(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_variant_a, task_description)
        evaluation = self.evaluate_click(result.get('last_coordinates'), SUBMIT_BBOX_TOP_RIGHT)
        return {"test_name": "variant_a", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_variant_b(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_variant_b, task_description)
        evaluation = self.evaluate_click(result.get('last_coordinates'), SUBMIT_BBOX_BOTTOM_CENTER)
        return {"test_name": "variant_b", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task_description: str) -> Dict[str, Any]:
        a = await self.run_variant_a(task_description)
        b = await self.run_variant_b(task_description)
        overall = a["passed"] and b["passed"]
        return {"overall_passed": overall, "variant_a": a, "variant_b": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_button_placement_variation():
    task_description = (
        """
        Click the "Submit" button to submit the form.
        Variant A: Button located at the top-right corner.
        Variant B: Button located at the bottom-center.
        """
    )
    t = ButtonPlacementVariationTest()
    results = await t.run_both(task_description)
    print("\nBUTTON PLACEMENT VARIATION RESULTS")
    for name in ("variant_a", "variant_b"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_button_placement_variation()


if __name__ == "__main__":
    asyncio.run(main())

