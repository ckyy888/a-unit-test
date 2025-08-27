import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from cua_test_runner_v2 import run_agent_test  # type: ignore


class AmazonColourDifferenceTest:
    def __init__(self):
        self.url = "https://www.amazon.com/s?k=owala&i=garden&crid=3AQWT9N9ROONA&sprefix=owala%2Cgarden%2C199&ref=nb_sb_noss_1"
        # Success URLs for each test
        self.first_success_url = "https://www.amazon.com/Owala-FreeSip-Insulated-Stainless-BPA-Free/dp/B0C59CLN29/ref=sr_1_2?crid=3AQWT9N9ROONA&dib=eyJ2IjoiMSJ9.Mzadg62IWLrl6BkCIu8PUzU4KlqWYEXLY7E9aLzj7NHUvpaPKgbedtD6XesDWQFIBuiUXRw_K6V4OgeM9OtHlO_oAcfYltlK63z2CrHTQLHkKGRUvsnPE0R2Jvp7G0jqfY8IvoIsYwuQ6a6ZOpIGl5rYYvyKWU7zOANDlqbOo_ItQNt4AleydLv9UAWLQjHecmH6PLDvFsBakHw20L92jxExo16JrT7x8dltU-cH3pSs4zqD8bY2-QiTxWB5w5eRyBVdG4s4T6y2uM38qviY_l4Zq8n-Txw4qImNLo8n460.IRHIJnNO7ntshKHTtbRnYlmmTnAjbfrnPWB-sEWa5oo&dib_tag=se&keywords=owala&qid=1756267903&s=home-garden&sprefix=owala%2Cgarden%2C199&sr=1-2&th=1"
        self.second_success_url = "https://www.amazon.com/Owala-FreeSip-Insulated-Stainless-BPA-Free/dp/B0C59BYVTQ/ref=sr_1_2?crid=3AQWT9N9ROONA&dib=eyJ2IjoiMSJ9.Mzadg62IWLrl6BkCIu8PUzU4KlqWYEXLY7E9aLzj7NHUvpaPKgbedtD6XesDWQFIBuiUXRw_K6V4OgeM9OtHlO_oAcfYltlK63z2CrHTQLHkKGRUvsnPE0R2Jvp7G0jqfY8IvoIsYwuQ6a6ZOpIGl5rYYvyKWU7zOANDlqbOo_ItQNt4AleydLv9UAWLQjHecmH6PLDvFsBakHw20L92jxExo16JrT7x8dltU-cH3pSs4zqD8bY2-QiTxWB5w5eRyBVdG4s4T6y2uM38qviY_l4Zq8n-Txw4qImNLo8n460.IRHIJnNO7ntshKHTtbRnYlmmTnAjbfrnPWB-sEWa5oo&dib_tag=se&keywords=owala&qid=1756267903&s=home-garden&sprefix=owala%2Cgarden%2C199&sr=1-2&th=1"
        self.first_task_description = (
            """
            You are a shopping assistant. Your task is to purchase an Owala FreeSip Insulated Stainless Steel Water Bottle for your user.
            Your user wants to purchase the colour "Iced Breeze". You have searched for owala in the amazon search bar.
            You plan to select the Owala FreeSip Insulated Stainless Steel Water Bottle. Once you have navigated to the product page, 
            and found the correct product,you must click the "Add to Cart" button.
            """
        )
        self.second_task_description = (
            """
            You are a shopping assistant. Your task is to purchase an Owala FreeSip Insulated Stainless Steel Water Bottle for your user.
            Your user wants to purchase the colour "Dreamy Field". You have searched for owala in the amazon search bar.
            You plan to select the Owala FreeSip Insulated Stainless Steel Water Bottle. Once you have navigated to the product page, 
            and found the correct product, you must click the "Add to Cart" button.
            """
        )

    def evaluate_purchase(self, agent_result: Dict[str, Any], expected_url: str) -> Dict[str, Any]:
        final_url = agent_result.get("final_url", "")
        passed = final_url == expected_url
        return {
            "passed": passed,
            "reason": (
                "Correct colour selected and navigated to correct product page"
                if passed
                else f"Incorrect URL reached. Expected the specific product page for the requested colour."
            ),
            "expected": expected_url,
            "actual": final_url,
        }

    async def run_first_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = self.evaluate_purchase(result, self.first_success_url)
        return {
            "test_name": "first_test",
            "scenario": "colour_label_standard",
            "agent_result": result,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }

    async def run_second_test(self, task_description: str) -> Dict[str, Any]:
        result = await run_agent_test(self.url, task_description)
        evaluation = self.evaluate_purchase(result, self.second_success_url)
        return {
            "test_name": "second_test",
            "scenario": "colour_label_variant",
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


async def test_amazon_colour_difference():
    test_runner = AmazonColourDifferenceTest()
    results = await test_runner.run_both_tests(
        test_runner.first_task_description, test_runner.second_task_description
    )

    print("\n" + "=" * 50)
    print("AMAZON COLOUR DIFFERENCE TEST RESULTS")
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
    return await test_amazon_colour_difference()


if __name__ == "__main__":
    asyncio.run(main())

