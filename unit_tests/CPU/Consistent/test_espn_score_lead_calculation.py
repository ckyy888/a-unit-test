#LATER.
import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cua_test_runner_v2 import run_agent_test  # type: ignore
from test_logger import TestLogger, get_test_name_from_file


class ESPNScoreLeadCalculationTest:
    def __init__(self):
        self.first_url = "https://www.espn.com/nba/story/_/id/45017098/minnesota-timberwolves-golden-state-warriors-game-1-nba-playoffs-2025-updates-highlights-results"
        self.second_url = "https://www.espn.com/nba/game/_/gameId/401769837/warriors-timberwolves"
        self.task_description = (
            """
            Report who the leading team is.
            """
        )

    def extract_agent_final_output(self, agent_result: Dict[str, Any]) -> str:
        """Extract the agent's final text output/reasoning from the result."""
        result = agent_result.get("result", {})
        if not result or "output" not in result:
            return ""
        
        # Get the last reasoning output
        final_reasoning = ""
        for item in result["output"]:
            if item.get("type") == "reasoning":
                final_reasoning = item.get("output_text", "")
        
        return final_reasoning

    def evaluate_summary(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        agent_output = self.extract_agent_final_output(agent_result)
        
        # Check if the agent identified the leading team correctly
        # Look for keywords indicating the agent reported the leading team
        leading_keywords = ["leading", "lead", "ahead", "winning", "leading team", "in the lead"]
        team_keywords = ["warriors", "timberwolves", "golden state", "minnesota"]
        
        has_leading_info = any(keyword.lower() in agent_output.lower() for keyword in leading_keywords)
        has_team_info = any(keyword.lower() in agent_output.lower() for keyword in team_keywords)
        
        passed = has_leading_info and has_team_info
        
        return {
            "passed": passed,
            "reason": (
                "Agent successfully identified and reported the leading team"
                if passed
                else f"Agent output did not clearly identify the leading team. Output: {agent_output[:200]}..."
            ),
            "expected": "Agent should identify and report which team is leading",
            "actual": agent_output[:200] + "..." if len(agent_output) > 200 else agent_output,
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.first_url, task_description)
        evaluation = self.evaluate_summary(result)
        return {
            "test_name": "first_test",
            "scenario": "score_view",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.second_url, task_description)
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
        test_runner.task_description, test_runner.task_description
    )

    # Initialize logger and log results to JSON
    logger = TestLogger("espn_score_lead_calculation")
    logger.log_and_print_summary(results)
    
    return results


async def main():
    return await test_espn_score_lead_calculation()


if __name__ == "__main__":
    asyncio.run(main())

