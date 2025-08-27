import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cua_test_runner_v2 import run_agent_test  # type: ignore


class EspnScoresTest:
    def __init__(self):
        self.url = "https://www.espn.com/nba/scoreboard/_/date/20250506"
        self.first_task_description = (
            """
            You are a helpful assistant that reports the scores of all NBA games that
            happen on a given day. You are currently on the scores page for May 6th, 
            2025. Once you have found all the scores for the day, you should stop and report. 
            """
        )
        self.second_task_description = (
            """
            You are a helpful assistant that reports the scores of all NBA games that
            happen on a given day. Your user hates the Timberwolves, and never wants
            to see any scores for them. You are currently on the scores page for May 6th, 
            2025. Once you have found all the scores for the day, you should stop and report.
            """
        )

    def evaluate_first_submission(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        # First agent should include all teams: Pacers, Cavaliers, Warriors, and Timberwolves
        response_text = agent_result.get("response", "").lower()
        required_teams = ["pacers", "cavaliers", "warriors", "timberwolves"]
        
        found_teams = []
        missing_teams = []
        
        for team in required_teams:
            if team in response_text:
                found_teams.append(team.title())
            else:
                missing_teams.append(team.title())
        
        passed = len(missing_teams) == 0
        
        return {
            "passed": passed,
            "reason": (
                f"All required teams found: {', '.join(found_teams)}" if passed 
                else f"Missing teams: {', '.join(missing_teams)}"
            ),
            "expected": "Response should include: Pacers, Cavaliers, Warriors, and Timberwolves",
            "actual": f"Found teams: {', '.join(found_teams)}",
            "found_teams": found_teams,
            "missing_teams": missing_teams,
        }

    def evaluate_second_submission(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        # Second agent should include ONLY Pacers and Cavaliers, and NOT Warriors or Timberwolves
        response_text = agent_result.get("response", "").lower()
        allowed_teams = ["pacers", "cavaliers"]
        forbidden_teams = ["warriors", "timberwolves"]
        
        found_allowed = []
        found_forbidden = []
        missing_allowed = []
        
        for team in allowed_teams:
            if team in response_text:
                found_allowed.append(team.title())
            else:
                missing_allowed.append(team.title())
        
        for team in forbidden_teams:
            if team in response_text:
                found_forbidden.append(team.title())
        
        # Pass if all allowed teams are found AND no forbidden teams are found
        passed = len(missing_allowed) == 0 and len(found_forbidden) == 0
        
        if not passed:
            if missing_allowed and found_forbidden:
                reason = f"Missing allowed teams: {', '.join(missing_allowed)}; Found forbidden teams: {', '.join(found_forbidden)}"
            elif missing_allowed:
                reason = f"Missing allowed teams: {', '.join(missing_allowed)}"
            elif found_forbidden:
                reason = f"Found forbidden teams: {', '.join(found_forbidden)}"
            else:
                reason = "Unknown evaluation error"
        else:
            reason = f"Correctly found only allowed teams: {', '.join(found_allowed)}"
        
        return {
            "passed": passed,
            "reason": reason,
            "expected": "Response should include ONLY Pacers and Cavaliers (no Warriors or Timberwolves)",
            "actual": f"Found allowed: {', '.join(found_allowed)}; Found forbidden: {', '.join(found_forbidden)}",
            "found_allowed": found_allowed,
            "found_forbidden": found_forbidden,
            "missing_allowed": missing_allowed,
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = self.evaluate_first_submission(result)
        return {
            "test_name": "first_test",
            "scenario": "all_teams_included",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = self.evaluate_second_submission(result)
        return {
            "test_name": "second_test",
            "scenario": "filtered_teams_only",
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


async def test_espn_scores():
    test_runner = EspnScoresTest()
    results = await test_runner.run_both_tests(
        test_runner.first_task_description, test_runner.second_task_description
    )

    print("\n" + "=" * 50)
    print("ESPN SCORES TEST RESULTS")
    print("=" * 50)
    for test_name, test_result in [
        ("First Test (All Teams)", results["first_test"]),
        ("Second Test (Filtered Teams)", results["second_test"]),
    ]:
        print(f"\n{test_name}:")
        print(f"  Status: {'PASSED' if test_result['passed'] else 'FAILED'}")
        print(f"  Reason: {test_result['evaluation']['reason']}")
        print(f"  Expected: {test_result['evaluation']['expected']}")
        print(f"  Actual: {test_result['evaluation']['actual']}")

    print(
        f"\nOverall Result: {'PASSED' if results['overall_passed'] else 'FAILED'}"
    )
    print(
        f"Tests passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}"
    )
    return results


async def main():
    return await test_espn_scores()


if __name__ == "__main__":
    asyncio.run(main())

