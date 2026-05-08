"""
Reproducible demo of the MosquitoControl environment.
Runs a random agent and prints step-by-step results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MosquitoAction
from server.environment import MosquitoControlEnvironment, ACTION_TYPES, GRID_SIZE, TASKS


def run_demo():
    print("=" * 60)
    print("MosquitoControl OpenEnv — Demo")
    print("=" * 60)

    env = MosquitoControlEnvironment()
    obs = env.reset(seed=42, task_id="easy")

    print(f"\nTask: {obs.task_id} ({obs.difficulty})")
    print(f"Budget: {obs.budget:.1f}  Max Steps: {obs.max_steps}")
    print(f"Weather: rain={obs.rainfall:.2f} temp={obs.temperature:.1f} hum={obs.humidity:.2f}")
    print(f"Mosquito: {obs.total_mosquito:.3f}  Infection: {obs.total_infection:.3f}  Hotspots: {obs.hotspot_count}")
    print()

    total_reward = 0.0
    while not obs.done:
        a_type = env.rng.choice(ACTION_TYPES)
        action = MosquitoAction(
            action_type=a_type,
            x=int(env.rng.randint(0, GRID_SIZE)),
            y=int(env.rng.randint(0, GRID_SIZE)),
        )
        obs = env.step(action)
        total_reward += obs.reward

        print(
            f"Step {obs.step_count:2d} | "
            f"{action.action_type:10s} ({action.x},{action.y}) | "
            f"Reward: {obs.reward:+7.2f} | "
            f"Budget: {obs.budget:5.1f} | "
            f"Inf: {obs.total_infection:6.2f} | "
            f"Mosq: {obs.total_mosquito:6.2f} | "
            f"Hot: {obs.hotspot_count:2d}"
        )

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Budget remaining: {env.budget:.1f}")

    # Run graders
    print("\n" + "=" * 60)
    print("Grader Results")
    print("=" * 60)
    for tid in TASKS:
        result = MosquitoControlEnvironment.run_grader(tid)
        passed = "PASS" if result["passed"] else "FAIL"
        print(f"  {tid:8s}  score={result['score']:.4f}  {passed}")


if __name__ == "__main__":
    run_demo()
