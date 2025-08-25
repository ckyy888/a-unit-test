import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


CLUSTER_BBOX = {"x": 1100, "y": 550, "width": 300, "height": 300}
PIN_BBOX = {"x": 1350, "y": 600, "width": 120, "height": 120}


class GoogleMapsZoomPrecisionTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.good_zoom_path = os.path.join(self.screenshots_dir, 'good.png')
        self.bad_zoom_path = os.path.join(self.screenshots_dir, 'bad.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (1920, 1080), color='white')

    def get_good(self):
        return self._get(self.good_zoom_path)

    def get_bad(self):
        return self._get(self.bad_zoom_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]], bbox: Dict[str, float], desc: str) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No interaction detected", "expected": desc, "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, bbox)
        return {"passed": ok, "reason": f"Interaction ({x},{y}) {'inside' if ok else 'outside'} target region", "expected": desc, "actual": f"({x},{y})"}

    async def run_good(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_good, task)
        evaluation = self.evaluate(result.get('last_coordinates'), PIN_BBOX, "Click individual coffee shop pin")
        return {"test_name": "good_zoom", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_bad(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_bad, task)
        evaluation = self.evaluate(result.get('last_coordinates'), CLUSTER_BBOX, "Adjust zoom/pan to separate cluster")
        return {"test_name": "bad_zoom", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_good(task)
        b = await self.run_bad(task)
        return {"overall_passed": a["passed"] and b["passed"], "good": a, "bad": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_google_maps_zoom_precision():
    task = (
        """
        Search "coffee shops near Stanford University" and interact with pins.
        If clustered, adjust zoom/pan until pins are distinct.
        """
    )
    t = GoogleMapsZoomPrecisionTest()
    results = await t.run_both(task)
    print("\nGOOGLE MAPS ZOOM PRECISION RESULTS")
    for name in ("good", "bad"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_google_maps_zoom_precision()


if __name__ == "__main__":
    asyncio.run(main())

