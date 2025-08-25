import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


ENROLL_BBOX = {"x": 1600, "y": 980, "width": 320, "height": 80}


class CourseraEnrollButtonContrastTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.high_path = os.path.join(self.screenshots_dir, 'high_contrast.png')
        self.low_path = os.path.join(self.screenshots_dir, 'low_contrast.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (1920, 1080), color='white')

    def get_high(self):
        return self._get(self.high_path)

    def get_low(self):
        return self._get(self.low_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No click detected", "expected": "Click Enroll", "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, ENROLL_BBOX)
        return {"passed": ok, "reason": f"Click ({x},{y}) {'inside' if ok else 'outside'} Enroll", "expected": "Click Enroll", "actual": f"({x},{y})"}

    async def run_high(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_high, task)
        evaluation = self.evaluate(result.get('last_coordinates'))
        return {"test_name": "high_contrast", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_low(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_low, task)
        evaluation = self.evaluate(result.get('last_coordinates'))
        return {"test_name": "low_contrast", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_high(task)
        b = await self.run_low(task)
        return {"overall_passed": a["passed"] and b["passed"], "high": a, "low": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_coursera_enroll_button_contrast():
    task = (
        """
        Click the "Enroll" button to enroll regardless of contrast level.
        """
    )
    t = CourseraEnrollButtonContrastTest()
    results = await t.run_both(task)
    print("\nCOURSERA ENROLL BUTTON CONTRAST RESULTS")
    for name in ("high", "low"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_coursera_enroll_button_contrast()


if __name__ == "__main__":
    asyncio.run(main())

