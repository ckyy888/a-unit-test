import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cua_test_runner_v2 import run_agent_test  # type: ignore


class AmazonProductSpecificationSequenceTest:
    def __init__(self):
        self.url = "https://www.amazon.com/Hoka-Womens-Clifton-Sneaker-White/dp/B0D5G7P5XS/ref=sr_1_1?crid=22EA8AUPQ36SO&dib=eyJ2IjoiMSJ9.1wswctvvq20XfWiP02Et43JU1SJ0047FmCYFLWjQ0gNpUvPiDRScPBrUkpuG5cVvbhKoORIJL8muKSUHhI7mVnsqRAOhYELjBbY3uKd3TtCR5n6_Y0jHxp238iWHzijpLrKBURFvfA7U5pjU6FtZ_uo_2zNWFkK0Mb690xCYGb-5ScR-duWRpA5TLI79nydpvWxUOYg47MVCAJKC5V1i5klXpQx8TWJCyjHvRYrXUaBsxQtguuUmjg7ybX1uAfzngM4ObVA1N0Z5KW32aN9q8A4QFVwp8rjsgUv3KTVglDY.hptID6-aPJObatYcETb_DLtwSpygvxrA6Ec_ioemtEM&dib_tag=se&keywords=hokas&qid=1756270130&sprefix=hoka%2Caps%2C220&sr=8-1"
        self.first_task_description = (
            """
            You are a shopping assistant. Your task is to purchase a Hoka Womens Clifton 10 Sneaker for your user.
            Your user wants to purchase them in black. She wears a size 8.5. You previously searched for hokas in the amazon 
            search bar and are now on the product details page. You plan to select the correct colour and size. 
            Terminate the task once you have done that.
            """
        )
        self.second_task_description = (
            """
            You are a shopping assistant. Your task is to purchase a Hoka Womens Clifton 10 Sneaker for your user.
            Your user wears a size 8.5. She wants to purchase them in black. You previously searched for hokas in the amazon 
            search bar and are now on the product details page. You plan to select the correct size and colour. 
            Terminate the task once you have done that.
            """
        )

    def evaluate_decision(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        final_url = agent_result.get("final_url", "")
        expected_url = "https://www.amazon.com/Hoka-Womens-Clifton-Sneaker-White/dp/B0D5G1525Y/ref=sr_1_1?crid=22EA8AUPQ36SO&dib=eyJ2IjoiMSJ9.1wswctvvq20XfWiP02Et43JU1SJ0047FmCYFLWjQ0gNpUvPiDRScPBrUkpuG5cVvbhKoORIJL8muKSUHhI7mVnsqRAOhYELjBbY3uKd3TtCR5n6_Y0jHxp238iWHzijpLrKBURFvfA7U5pjU6FtZ_uo_2zNWFkK0Mb690xCYGb-5ScR-duWRpA5TLI79nydpvWxUOYg47MVCAJKC5V1i5klXpQx8TWJCyjHvRYrXUaBsxQtguuUmjg7ybX1uAfzngM4ObVA1N0Z5KW32aN9q8A4QFVwp8rjsgUv3KTVglDY.hptID6-aPJObatYcETb_DLtwSpygvxrA6Ec_ioemtEM&dib_tag=se&keywords=hokas&qid=1756270130&sprefix=hoka%2Caps%2C220&sr=8-1&th=1&psc=1"
        passed = final_url == expected_url
        return {
            "passed": passed,
            "reason": (
                "Reached expected Amazon product page with correct configuration"
                if passed
                else f"Final URL does not match expected URL. Expected: {expected_url}, Got: {final_url}"
            ),
            "expected": expected_url,
            "actual": final_url,
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = self.evaluate_decision(result)
        return {
            "test_name": "first_test",
            "scenario": "sequence_brand_first",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = self.evaluate_decision(result)
        return {
            "test_name": "second_test",
            "scenario": "sequence_reviews_first",
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


async def test_amazon_product_specification_sequence():
    test_runner = AmazonProductSpecificationSequenceTest()
    results = await test_runner.run_both_tests(
        test_runner.first_task_description, test_runner.second_task_description
    )

    print("\n" + "=" * 50)
    print("AMAZON PRODUCT SPECIFICATION SEQUENCE TEST RESULTS")
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
    return await test_amazon_product_specification_sequence()


if __name__ == "__main__":
    asyncio.run(main())

