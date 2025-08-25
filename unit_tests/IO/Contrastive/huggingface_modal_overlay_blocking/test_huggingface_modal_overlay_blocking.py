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
MODAL_DISMISS_BBOX = {"x": 960, "y": 540, "width": 120, "height": 60}


class HuggingfaceModalOverlayBlockingTest:
    def __init__(self):
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        self.clean_path = os.path.join(self.screenshots_dir, 'clean.png')
        self.modal_path = os.path.join(self.screenshots_dir, 'modal.png')

    def _get(self, path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception:
            return Image.new('RGB', (1920, 1080), color='white')

    def get_clean(self):
        return self._get(self.clean_path)

    def get_modal(self):
        return self._get(self.modal_path)

    def _inside(self, x: int, y: int, bbox: Dict[str, float]) -> bool:
        return bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]

    def evaluate(self, coords: Optional[Tuple[int, int]], bbox: Dict[str, float], desc: str) -> Dict[str, Any]:
        if coords is None:
            return {"passed": False, "reason": "No click detected", "expected": desc, "actual": "None"}
        x, y = coords
        ok = self._inside(x, y, bbox)
        return {"passed": ok, "reason": f"Click ({x},{y}) {'inside' if ok else 'outside'} expected region", "expected": desc, "actual": f"({x},{y})"}

    async def run_clean(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_clean, task)
        evaluation = self.evaluate(result.get('last_coordinates'), DOWNLOAD_BBOX, "Click Download")
        return {"test_name": "clean_page", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_modal(self, task: str) -> Dict[str, Any]:
        result = await run_agent_test(self.get_modal, task)
        evaluation = self.evaluate(result.get('last_coordinates'), MODAL_DISMISS_BBOX, "Dismiss cookie modal first")
        return {"test_name": "modal_blocking", "agent_result": result, "evaluation": evaluation, "passed": evaluation["passed"]}

    async def run_both(self, task: str) -> Dict[str, Any]:
        a = await self.run_clean(task)
        b = await self.run_modal(task)
        return {"overall_passed": a["passed"] and b["passed"], "clean": a, "modal": b, "summary": {"total_tests": 2, "passed_tests": sum([a["passed"], b["passed"]])}}


async def test_huggingface_modal_overlay_blocking():
    task = (
        """
        Access the model download. If a cookie consent modal overlays it, dismiss the modal first.
        Then click the Download button.
        """
    )
    t = HuggingfaceModalOverlayBlockingTest()
    results = await t.run_both(task)
    print("\nHUGGINGFACE MODAL OVERLAY BLOCKING RESULTS")
    for name in ("clean", "modal"):
        tr = results[name]
        print(f"  {name}: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['evaluation']['reason']}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    return results


async def main():
    return await test_huggingface_modal_overlay_blocking()


if __name__ == "__main__":
    asyncio.run(main())

