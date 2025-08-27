import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cua_test_runner_v2 import run_agent_test  # type: ignore


class AllRecipesServingSizeContextTest:
    def __init__(self):
        self.first_url = "https://www.allrecipes.com/recipe/20144/banana-banana-bread/"
        self.second_url = "https://www.allrecipes.com/recipe/17066/janets-rich-banana-bread/"
        
        self.task_description = (
            """
            You are a helpful assistant that has been tasked with finding a recipe for banana bread.
            Your user wants to make a recipe that serves 12 people. You previously randomly selected
            a recipe and you are on that page. Check if the user's requirements are met. If they are,
            stop the search. If not, go back to the page showing all available banana bread recipes.
            """
        )

    def evaluate_first_test(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        final_url = agent_result.get("final_url", "")
        final_reasoning = agent_result.get("final_reasoning", "")
        
        # First test passes if agent stops and confirms they found the correct recipe
        # This means they should stay on the first_url and indicate they found what they need
        passed = (
            final_url == self.first_url and (
                "correct recipe" in final_reasoning.lower() or
                "found" in final_reasoning.lower() or
                "serves 12" in final_reasoning.lower() or
                "requirement" in final_reasoning.lower()
            )
        )
        
        return {
            "passed": passed,
            "reason": (
                "Agent correctly identified the recipe meets requirements and stopped" 
                if passed else 
                f"Agent did not confirm correct recipe found. Final URL: {final_url}, Reasoning: {final_reasoning}"
            ),
            "expected": f"Stay on {self.first_url} and confirm recipe is correct",
            "actual": f"URL: {final_url}, Reasoning: {final_reasoning}",
        }

    def evaluate_second_test(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        final_url = agent_result.get("final_url", "")
        
        # Second test passes if final URL is different from second_url
        # This means the agent recognized the recipe doesn't serve 12 and navigated away
        passed = final_url != self.second_url
        
        return {
            "passed": passed,
            "reason": (
                "Agent correctly recognized recipe doesn't serve 12 and navigated away" 
                if passed else 
                f"Agent stayed on the page that doesn't meet requirements: {final_url}"
            ),
            "expected": f"Navigate away from {self.second_url}",
            "actual": final_url,
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.first_url, task_description)
        evaluation = self.evaluate_first_test(result)
        return {
            "test_name": "first_test",
            "scenario": "recipe_serves_12_should_stop",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.second_url, task_description)
        evaluation = self.evaluate_second_test(result)
        return {
            "test_name": "second_test",
            "scenario": "recipe_doesnt_serve_12_should_navigate_away",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_both_tests(self) -> Dict[str, Any]:
        first_test_result = await self.run_first_test(self.task_description)
        second_test_result = await self.run_second_test(self.task_description)
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


async def test_allrecipes_serving_size_context():
    test_runner = AllRecipesServingSizeContextTest()
    results = await test_runner.run_both_tests()

    print("\n" + "=" * 50)
    print("ALLRECIPES SERVING SIZE CONTEXT TEST RESULTS")
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
    return await test_allrecipes_serving_size_context()


if __name__ == "__main__":
    asyncio.run(main())

