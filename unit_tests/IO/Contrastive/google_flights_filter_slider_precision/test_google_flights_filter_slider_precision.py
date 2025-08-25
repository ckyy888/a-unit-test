import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


TEXT_INPUT_BBOX = {"x": 1700, "y": 220, "width": 160, "height": 60}
SLIDER_HANDLE_BBOX = {"x": 900, "y": 230, "width": 40, "height": 40}


class GoogleFlightsFilterSliderPrecisionTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.text_input_path = os.path.join(self.screenshots_dir, 'text_input.png')
        self.slider_path = os.path.join(self.screenshots_dir, 'slider.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (1920, 1080), color='white')

    def get_text_input(self):
        return self._get(self.text_input_path)

    def get_slider(self):
        return self._get(self.slider_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]], bbox: Dict[str, float], desc: str) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No interaction detected", "expected": desc, "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, bbox)
        return {"passed": ok, "reason": f"Interaction ({x},{y}) {'inside' if ok else 'outside'} expected region", "expected": desc, "actual": f"({x},{y})"}

    async def run_text_input(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_text_input, task)
        evaluation = self.evaluate(result.get('last_coordinates'), TEXT_INPUT_BBOX, "Type $300 in price input")
        return {"test_name": "text_input", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_slider(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_slider, task)
        evaluation = self.evaluate(result.get('last_coordinates'), SLIDER_HANDLE_BBOX, "Drag slider handle to $300")
        return {"test_name": "slider_drag", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_text_input(task)
        b = await self.run_slider(task)
        return {"overall_passed": a["passed"] and b["passed"], "text_input": a, "slider": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_google_flights_filter_slider_precision():
    task = (
        """
        Set the maximum price filter to $300. If a text box is available, type it.
        If only a slider is available, drag the handle precisely to $300.
        """
    )
    t = GoogleFlightsFilterSliderPrecisionTest()
    results = await t.run_both(task)
    print("\nGOOGLE FLIGHTS FILTER SLIDER PRECISION RESULTS")
    for name in ("text_input", "slider"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_google_flights_filter_slider_precision()


if __name__ == "__main__":
    asyncio.run(main())

