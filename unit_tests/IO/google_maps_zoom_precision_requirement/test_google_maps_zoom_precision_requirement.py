import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


# Expected interaction regions
BUILDING_BBOX_PERFECT = {"x": 1200, "y": 600, "width": 200, "height": 200}
ZOOM_CONTROL_BBOX = {"x": 1850, "y": 250, "width": 60, "height": 120}  # +/- controls area


class GoogleMapsZoomPrecisionRequirementTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.perfect_path = os.path.join(self.screenshots_dir, 'perfect.png')
        self.street_level_path = os.path.join(self.screenshots_dir, 'street_level.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (1920, 1080), color='white')

    def get_perfect(self):
        return self._get(self.perfect_path)

    def get_street_level(self):
        return self._get(self.street_level_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]], bbox: Dict[str, float], expected_desc: str) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No interaction detected", "expected": expected_desc, "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, bbox)
        return {"passed": ok, "reason": f"Interaction ({x},{y}) {'inside' if ok else 'outside'} expected region", "expected": expected_desc, "actual": f"({x},{y})"}

    async def run_perfect(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_perfect, task)
        evaluation = self.evaluate(result.get('last_coordinates'), BUILDING_BBOX_PERFECT, "Click building marker")
        return {"test_name": "perfect_zoom", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_street_level(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_street_level, task)
        evaluation = self.evaluate(result.get('last_coordinates'), ZOOM_CONTROL_BBOX, "Adjust zoom precisely using controls")
        return {"test_name": "street_level_adjust", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_perfect(task)
        b = await self.run_street_level(task)
        return {"overall_passed": a["passed"] and b["passed"], "perfect": a, "street_level": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_google_maps_zoom_precision_requirement():
    task = (
        """
        Search the address and ensure the building is clearly visible.
        If already perfect, click the building marker. If not, adjust zoom precisely.
        """
    )
    t = GoogleMapsZoomPrecisionRequirementTest()
    results = await t.run_both(task)
    print("\nGOOGLE MAPS ZOOM PRECISION REQUIREMENT RESULTS")
    for name in ("perfect", "street_level"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_google_maps_zoom_precision_requirement()


if __name__ == "__main__":
    asyncio.run(main())

