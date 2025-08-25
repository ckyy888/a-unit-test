import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from cua_test_runner import run_agent_test  # type: ignore


DOWNLOAD_BBOX = {"x": 1700, "y": 400, "width": 300, "height": 80}


class HuggingfaceThemeVariationTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.light_path = os.path.join(self.screenshots_dir, 'light.png')
        self.dark_path = os.path.join(self.screenshots_dir, 'dark.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (1920, 1080), color='white')

    def get_light(self):
        return self._get(self.light_path)

    def get_dark(self):
        return self._get(self.dark_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No click detected", "expected": "Click Download", "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, DOWNLOAD_BBOX)
        return {"passed": ok, "reason": f"Click ({x},{y}) {'inside' if ok else 'outside'} Download", "expected": "Click Download", "actual": f"({x},{y})"}

    async def run_light(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_light, task)
        evaluation = self.evaluate(result.get('last_coordinates'))
        return {"test_name": "light", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_dark(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_dark, task)
        evaluation = self.evaluate(result.get('last_coordinates'))
        return {"test_name": "dark", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_light(task)
        b = await self.run_dark(task)
        return {"overall_passed": a["passed"] and b["passed"], "light": a, "dark": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_huggingface_theme_variation():
    task = (
        """
        Click the Download button for the model regardless of light or dark theme.
        """
    )
    t = HuggingfaceThemeVariationTest()
    results = await t.run_both(task)
    print("\nHUGGINGFACE THEME VARIATION RESULTS")
    for name in ("light", "dark"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_huggingface_theme_variation()


if __name__ == "__main__":
    asyncio.run(main())

