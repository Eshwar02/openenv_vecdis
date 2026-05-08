"""
tasks.py — Standalone task runner and grader for MosquitoControl.
Uses the static run_grader method from the environment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import MosquitoControlEnvironment, TASKS


def run_all_tasks():
    """Run all task graders and print results."""
    print("=" * 60)
    print("MosquitoControl OpenEnv — Task Graders")
    print("=" * 60)

    total_score = 0.0
    for task_id in TASKS:
        result = MosquitoControlEnvironment.run_grader(task_id)
        passed = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {task_id:8s} ({TASKS[task_id]['difficulty']:6s})  "
            f"score={result['score']:.4f}  {passed}  "
            f"{result['feedback']}"
        )
        total_score += result["score"]

    avg = total_score / len(TASKS)
    print(f"\n  Average score: {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tasks()
