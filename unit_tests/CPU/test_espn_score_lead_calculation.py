import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cua_test_runner_v2 import run_agent_test  # type: ignore


class ESPNScoreLeadCalculationTest:
    def __init__(self):
        self.url = "https://example.com/espn/game/live"
        self.first_task_description = (
            """
            Agent 1: You see the score displayed as "Lakers 108, Warriors 95".
            Task: Report the leading team and proceed to the summary page.
            """
        )
        self.second_task_description = (
            """
            Agent 2: You see the lead displayed as "Lakers lead by 13".
            Task: Report the leading team consistently and proceed to the summary page.
            """
        )

    def evaluate_summary(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        final_url = agent_result.get("final_url", "")
        success_prefix = "https://example.com/espn/game/summary"
        passed = final_url.startswith(success_prefix)
        return {
            "passed": passed,
            "reason": (
                "Reached summary page after consistent lead reasoning"
                if passed
                else f"Unexpected final URL: {final_url}"
            ),
            "expected": f"Final URL starts with {success_prefix}",
            "actual": final_url,
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = self.evaluate_summary(result)
        return {
            "test_name": "first_test",
            "scenario": "score_view",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = self.evaluate_summary(result)
        return {
            "test_name": "second_test",
            "scenario": "lead_view",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_both_tests(self, first_task_description: str, second_task_description: str) -> Dict[str, Any]:
        first_test_result = await self.run_first_test(first_task_description)
        second_test_result = await self.run_second_test(second_task_description)
        overall_passed = first_test_result["passed"] and second_test_result["passed"]
        return {
            "overall_passed": overall_passed,
            "first_test": first_test_result,
            "second_test": second_test_result,
            "summary": {
                "first_test_passed": first_test_result["passed"],
                "second_test_passed": second_test_result["passed"],
                "total_tests": 2,
                "passed_tests": sum([
                    first_test_result["passed"],
                    second_test_result["passed"],
                ]),
            },
        }


async def test_espn_score_lead_calculation():
    test_runner = ESPNScoreLeadCalculationTest()
    results = await test_runner.run_both_tests(
        test_runner.first_task_description, test_runner.second_task_description
    )

    print("\n" + "=" * 50)
    print("ESPN SCORE LEAD CALCULATION TEST RESULTS")
    print("=" * 50)
    for test_name, test_result in [
        ("First Test", results["first_test"]),
        ("Second Test", results["second_test"]),
    ]:
        print(f"\n{test_name}:")
        print(f"  Status: {'PASSED' if test_result['passed'] else 'FAILED'}")
        print(f"  Reason: {test_result['evaluation']['reason']}")
        final_url = test_result["evaluation"].get("actual", "Unknown")
        print(f"  Final URL: {final_url}")

    print(
        f"\nOverall Result: {'PASSED' if results['overall_passed'] else 'FAILED'}"
    )
    print(
        f"Tests passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}"
    )
    return results


async def main():
    return await test_espn_score_lead_calculation()


if __name__ == "__main__":
    asyncio.run(main())

