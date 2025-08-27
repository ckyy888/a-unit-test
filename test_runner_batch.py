#!/usr/bin/env python3
"""
Batch test runner for unit tests.
Runs each specified test file 5 times and logs pass/fail results to JSON.
"""

import asyncio
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Test file configurations with their relative paths from workspace root
TEST_CONFIGS = [
    {
        "name": "test_huggingface_leaderboard",
        "path": "unit_tests/CPU/Consistent/test_huggingface_leaderboard/test_huggingface_leaderboard.py",
        "working_dir": "unit_tests/CPU/Consistent/test_huggingface_leaderboard"
    },
    {
        "name": "test_booking_price_currency_display", 
        "path": "unit_tests/CPU/Consistent/test_booking_price_currency_display.py",
        "working_dir": "unit_tests/CPU/Consistent"
    },
    {
        "name": "test_espn_score_lead_calculation",
        "path": "unit_tests/CPU/Consistent/test_espn_score_lead_calculation.py", 
        "working_dir": "unit_tests/CPU/Consistent"
    },
    {
        "name": "test_amazon_price_difference",
        "path": "unit_tests/CPU/Contrastive/amazon_price_difference/test_amazon_price_difference.py",
        "working_dir": "unit_tests/CPU/Contrastive/amazon_price_difference"
    },
    {
        "name": "test_amazon_shipping",
        "path": "unit_tests/CPU/Contrastive/test_amazon_shipping.py",
        "working_dir": "unit_tests/CPU/Contrastive"
    },
    {
        "name": "test_course_era_careers",
        "path": "unit_tests/CPU/Contrastive/test_course_era_careers.py",
        "working_dir": "unit_tests/CPU/Contrastive"
    },
    {
        "name": "test_gflight_time_difference",
        "path": "unit_tests/CPU/Contrastive/test_gflight_time_difference.py",
        "working_dir": "unit_tests/CPU/Contrastive"
    },
    {
        "name": "test_amazon_size_colour",
        "path": "unit_tests/Memory/Consistent/test_amazon_size_colour/test_amazon_size_colour.py",
        "working_dir": "unit_tests/Memory/Consistent/test_amazon_size_colour"
    },
    {
        "name": "test_gmail_password_difference",
        "path": "unit_tests/Memory/Consistent/test_gmail_password_difference/test_gmail_password_difference.py",
        "working_dir": "unit_tests/Memory/Consistent/test_gmail_password_difference"
    },
    {
        "name": "test_leaking_password",
        "path": "unit_tests/Memory/Contrastive/test_leaking_password/test_leaking_password.py",
        "working_dir": "unit_tests/Memory/Contrastive/test_leaking_password"
    },
    {
        "name": "test_espn_scores",
        "path": "unit_tests/Memory/Contrastive/test_espn_scores.py",
        "working_dir": "unit_tests/Memory/Contrastive"
    },
    {
        "name": "test_gflight_religion",
        "path": "unit_tests/Memory/Contrastive/test_gflight_religion.py",
        "working_dir": "unit_tests/Memory/Contrastive"
    }
]

class BatchTestRunner:
    def __init__(self, workspace_root="/Users/charlotteyan/Desktop/unit-test"):
        self.workspace_root = Path(workspace_root)
        self.results = {}
        self.total_runs = 5
        
    def run_single_test(self, test_config, run_number):
        """Run a single test and return the results"""
        print(f"Running {test_config['name']} - Run {run_number + 1}/5")
        
        # Change to the test's working directory
        working_dir = self.workspace_root / test_config['working_dir']
        test_file = test_config['path'].split('/')[-1]  # Get just the filename
        
        try:
            # Run the test using subprocess from the correct directory
            result = subprocess.run(
                [sys.executable, test_file],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test
            )
            
            # Check if the process completed successfully
            if result.returncode == 0:
                # Try to parse any JSON output or success indicators
                output = result.stdout
                
                # Look for common success patterns in the output
                if "overall_passed" in output.lower() or "passed" in output.lower():
                    # Try to extract actual pass/fail from output if possible
                    if "overall_passed\": false" in output or "\"passed\": false" in output:
                        return False
                    elif "overall_passed\": true" in output or "\"passed\": true" in output:
                        return True
                    else:
                        # If we can't determine from output, assume success since returncode was 0
                        return True
                else:
                    # No clear success indicator, but process completed - assume success
                    return True
            else:
                print(f"Test {test_config['name']} failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"Test {test_config['name']} timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"Error running test {test_config['name']}: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests the specified number of times"""
        print(f"Starting batch test run - {self.total_runs} runs per test")
        print(f"Total tests to run: {len(TEST_CONFIGS) * self.total_runs}")
        
        for test_config in TEST_CONFIGS:
            test_name = test_config['name']
            print(f"\n{'='*60}")
            print(f"Running test: {test_name}")
            print(f"{'='*60}")
            
            passes = 0
            fails = 0
            
            for run_num in range(self.total_runs):
                try:
                    success = self.run_single_test(test_config, run_num)
                    if success:
                        passes += 1
                        print(f"  Run {run_num + 1}: PASS")
                    else:
                        fails += 1
                        print(f"  Run {run_num + 1}: FAIL")
                except Exception as e:
                    fails += 1
                    print(f"  Run {run_num + 1}: ERROR - {e}")
            
            self.results[test_name] = {
                "passes": passes,
                "fails": fails,
                "total_runs": self.total_runs,
                "pass_rate": passes / self.total_runs * 100
            }
            
            print(f"Results for {test_name}: {passes} passes, {fails} fails ({passes/self.total_runs*100:.1f}% pass rate)")
    
    def save_results(self, output_file="batch_test_results.json"):
        """Save results to JSON file"""
        output_data = {
            "test_run_info": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(TEST_CONFIGS),
                "runs_per_test": self.total_runs,
                "total_executions": len(TEST_CONFIGS) * self.total_runs
            },
            "results": self.results,
            "summary": {
                "total_passes": sum(r["passes"] for r in self.results.values()),
                "total_fails": sum(r["fails"] for r in self.results.values()),
                "overall_pass_rate": sum(r["passes"] for r in self.results.values()) / (len(TEST_CONFIGS) * self.total_runs) * 100
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("BATCH TEST SUMMARY")
        print(f"{'='*60}")
        for test_name, results in self.results.items():
            print(f"{test_name:35} | {results['passes']:2d} passes | {results['fails']:2d} fails | {results['pass_rate']:5.1f}%")
        
        print(f"\nOverall: {output_data['summary']['total_passes']} passes, {output_data['summary']['total_fails']} fails")
        print(f"Overall pass rate: {output_data['summary']['overall_pass_rate']:.1f}%")

def main():
    """Main function to run the batch tests"""
    runner = BatchTestRunner()
    runner.run_all_tests()
    runner.save_results()

if __name__ == "__main__":
    main()