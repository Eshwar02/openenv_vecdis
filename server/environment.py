"""
server/environment.py — Core simulation for the Mosquito Control Environment.

A 10x10 city grid where an RL agent allocates mosquito-control
resources under budget, weather, and hotspot constraints.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from openenv.core.env_server import Environment

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import MosquitoAction, MosquitoObservation, MosquitoState

# ── Constants ────────────────────────────────────────────────────────
GRID_SIZE = 10
CHANNELS = 4  # mosquito, infection, water, population

ACTION_TYPES = ["spray", "larvicide", "traps", "bed_net", "inspect", "no_op"]
ACTION_COSTS: Dict[str, float] = {
    "spray": 10.0,
    "larvicide": 7.0,
    "traps": 6.0,
    "bed_net": 3.0,
    "inspect": 2.0,
    "no_op": 0.0,
}

# ── Task presets ─────────────────────────────────────────────────────
TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "name": "Outbreak Suppression",
        "description": "Reduce total infection below 5.0 within 50 steps. Full budget available.",
        "difficulty": "easy",
        "rainfall_range": (0.2, 0.5),
        "budget": 100.0,
        "max_steps": 50,
        "infection_threshold": 5.0,
    },
    "medium": {
        "name": "Budget Efficiency",
        "description": "Control outbreak (infection < 10.0) with only half the normal budget. Requires smart allocation.",
        "difficulty": "medium",
        "rainfall_range": (0.2, 0.5),
        "budget": 50.0,
        "max_steps": 50,
        "infection_threshold": 10.0,
    },
    "hard": {
        "name": "Monsoon Resilience",
        "description": "Survive a high-rainfall monsoon scenario. Aggressive mosquito breeding, keep infection < 15.0.",
        "difficulty": "hard",
        "rainfall_range": (0.6, 0.95),
        "budget": 120.0,
        "max_steps": 60,
        "infection_threshold": 15.0,
    },
}


class MosquitoControlEnvironment(
    Environment[MosquitoAction, MosquitoObservation, MosquitoState]
):
    """
    OpenEnv environment for adaptive mosquito control.

    The agent manages a 10x10 city grid. Each cell tracks:
      - mosquito density (0–1)
      - infection risk (0–1)
      - standing water level (0–1)
      - population density (0–1, static per episode)

    Every step, weather changes, mosquitoes breed and spread,
    infections propagate, and the agent picks one intervention.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def close(self) -> None:
        """No-op: singleton instance persists across HTTP requests."""
        pass

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.grid: Optional[np.ndarray] = None
        self.rng = np.random.RandomState(42)
        self.budget = 0.0
        self._step_count = 0
        self.max_steps = 50
        self.rainfall = 0.0
        self.temperature = 0.0
        self.humidity = 0.0
        self._done = False
        self.total_reward = 0.0
        self._episode_id = ""
        self.last_action: Optional[Dict[str, Any]] = None
        self._prev_infection_sum = 0.0
        self.task_id = "easy"
        self.difficulty = "easy"
        self.rainfall_range = (0.2, 0.5)
        self.infection_threshold = 5.0
        self.initial_budget = 100.0

    # ── reset ────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MosquitoObservation:
        self._reset_rubric()

        # Task selection
        task_id = kwargs.get("task_id", "easy")
        if task_id not in TASKS:
            task_id = "easy"
        task = TASKS[task_id]
        self.task_id = task_id
        self.difficulty = task["difficulty"]
        self.rainfall_range = task["rainfall_range"]
        self.initial_budget = task["budget"]
        self.max_steps = task["max_steps"]
        self.infection_threshold = task["infection_threshold"]

        # Seed
        actual_seed = seed if seed is not None else 42
        self.rng = np.random.RandomState(actual_seed)

        # Initialize grid
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE, CHANNELS), dtype=np.float64)
        self.grid[:, :, 0] = self.rng.uniform(0.05, 0.3, (GRID_SIZE, GRID_SIZE))
        self.grid[:, :, 1] = self.rng.uniform(0.0, 0.15, (GRID_SIZE, GRID_SIZE))
        self.grid[:, :, 2] = self.rng.uniform(0.1, 0.6, (GRID_SIZE, GRID_SIZE))
        self.grid[:, :, 3] = self.rng.uniform(0.2, 1.0, (GRID_SIZE, GRID_SIZE))

        # Seed hotspots
        for _ in range(3):
            hx, hy = self.rng.randint(0, GRID_SIZE, 2)
            self.grid[hx, hy, 0] = self.rng.uniform(0.6, 0.9)
            self.grid[hx, hy, 1] = self.rng.uniform(0.3, 0.6)

        self.budget = self.initial_budget
        self._step_count = 0
        self._done = False
        self.total_reward = 0.0
        self._episode_id = episode_id or uuid.uuid4().hex[:8]
        self.last_action = None

        self._update_weather()
        self._prev_infection_sum = float(self.grid[:, :, 1].sum())

        return self._make_obs(
            reward=0.0,
            feedback=f"Episode started. Task: {task['name']} ({self.difficulty}). "
                     f"Budget: {self.budget}. Target: infection < {self.infection_threshold}.",
        )

    # ── step ─────────────────────────────────────────────────────
    def step(
        self,
        action: MosquitoAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MosquitoObservation:
        if self.grid is None:
            raise RuntimeError("Must call reset() before step()")
        if self._done:
            return self._make_obs(reward=0.0, feedback="Episode already finished.")

        reward = 0.0
        cost = 0.0
        feedback_parts: List[str] = []

        # Validate action type
        a_type = action.action_type
        x, y = action.x, action.y

        if a_type not in ACTION_TYPES:
            reward -= 5.0
            feedback_parts.append(f"Invalid action '{a_type}'. Penalty -5.")
            a_type = "no_op"

        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            reward -= 5.0
            feedback_parts.append(f"Invalid target ({x},{y}). Grid is 0-9. Penalty -5.")
            a_type = "no_op"
            x, y = 0, 0

        cost = ACTION_COSTS[a_type]

        # Budget check
        if cost > self.budget:
            reward -= 3.0
            feedback_parts.append(
                f"Cannot afford {a_type} (cost {cost}, budget {self.budget:.1f}). Penalty -3."
            )
            a_type = "no_op"
            cost = 0.0

        self.budget -= cost

        # Apply action
        before_mosq = float(self.grid[x, y, 0]) if a_type != "no_op" else 0
        before_inf = float(self.grid[x, y, 1]) if a_type != "no_op" else 0
        self._apply_action(a_type, x, y)
        after_mosq = float(self.grid[x, y, 0]) if a_type != "no_op" else 0
        after_inf = float(self.grid[x, y, 1]) if a_type != "no_op" else 0

        if a_type != "no_op":
            feedback_parts.append(
                f"Applied {a_type} at ({x},{y}). Cost: {cost}. "
                f"Cell mosquito: {before_mosq:.3f}->{after_mosq:.3f}, "
                f"infection: {before_inf:.3f}->{after_inf:.3f}."
            )

        # Environment dynamics
        self._update_weather()
        self._mosquito_growth()
        self._infection_spread()
        self.grid = np.clip(self.grid, 0.0, 1.0)

        # Compute reward
        infection_sum = float(self.grid[:, :, 1].sum())
        mosquito_sum = float(self.grid[:, :, 0].sum())
        infection_reduction = max(0.0, self._prev_infection_sum - infection_sum)
        hotspot_cells = float(np.sum(self.grid[:, :, 0] > 0.5))
        hotspot_bonus = max(0.0, 5.0 - hotspot_cells) * 0.5

        reward += (
            -infection_sum * 0.5
            - mosquito_sum * 0.3
            - cost * 0.1
            + infection_reduction * 8.0
            + hotspot_bonus
        )
        reward = float(np.clip(reward, -50.0, 50.0))

        self._prev_infection_sum = infection_sum
        self.total_reward += reward
        self._step_count += 1
        self.last_action = {"action_type": a_type, "x": x, "y": y}

        feedback_parts.append(
            f"Weather: rain={self.rainfall:.2f}, temp={self.temperature:.1f}C, humid={self.humidity:.2f}. "
            f"Totals — mosquito: {mosquito_sum:.2f}, infection: {infection_sum:.2f}, "
            f"hotspots: {int(hotspot_cells)}, budget: {self.budget:.1f}."
        )

        # Done conditions
        if self._step_count >= self.max_steps:
            self._done = True
            feedback_parts.append("Max steps reached.")
        if self.budget <= 0:
            self._done = True
            feedback_parts.append("Budget exhausted.")
        if infection_sum < 0.5:
            self._done = True
            feedback_parts.append("Infection fully controlled! Great job.")

        return self._make_obs(reward=reward, feedback=" ".join(feedback_parts))

    # ── state property ───────────────────────────────────────
    @property
    def state(self) -> MosquitoState:
        return MosquitoState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self.task_id,
            difficulty=self.difficulty,
            budget=self.budget,
            max_steps=self.max_steps,
            total_reward=self.total_reward,
            total_infection=self._get_total_infection(),
            total_mosquito=self._get_total_mosquito(),
            done=self._done,
        )

    # ── Static methods for server endpoints ──────────────────
    @staticmethod
    def list_tasks() -> List[Dict[str, Any]]:
        """Return task metadata for the /tasks endpoint."""
        return [
            {
                "id": tid,
                "name": t["name"],
                "description": t["description"],
                "difficulty": t["difficulty"],
                "max_steps": t["max_steps"],
                "budget": t["budget"],
                "infection_threshold": t["infection_threshold"],
            }
            for tid, t in TASKS.items()
        ]

    @staticmethod
    def run_grader(task_id: str, actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run the grader for a task. If no actions provided, uses random agent.
        Returns dict with score (0.0–1.0), passed, and feedback.
        """
        if task_id not in TASKS:
            return {"score": 0.0, "passed": False, "feedback": f"Unknown task: {task_id}"}

        task = TASKS[task_id]
        env = MosquitoControlEnvironment()
        obs = env.reset(seed=42, task_id=task_id)

        if actions:
            for a in actions:
                if obs.done:
                    break
                act = MosquitoAction(
                    action_type=a.get("action_type", "no_op"),
                    x=a.get("x", 0),
                    y=a.get("y", 0),
                )
                obs = env.step(act)
        else:
            # Random agent baseline
            for _ in range(task["max_steps"]):
                if obs.done:
                    break
                act = MosquitoAction(
                    action_type=env.rng.choice(ACTION_TYPES),
                    x=int(env.rng.randint(0, GRID_SIZE)),
                    y=int(env.rng.randint(0, GRID_SIZE)),
                )
                obs = env.step(act)

        final_infection = env._get_total_infection()
        threshold = task["infection_threshold"]

        # Score: 0.0–1.0
        if final_infection < threshold:
            infection_score = 1.0
        elif final_infection < threshold * 4:
            infection_score = max(0.0, 1.0 - (final_infection - threshold) / (threshold * 3))
        else:
            infection_score = 0.0

        budget_score = max(0.0, env.budget / env.initial_budget)
        score = round(0.7 * infection_score + 0.3 * budget_score, 4)

        return {
            "score": score,
            "passed": score >= 0.6,
            "feedback": (
                f"Final infection: {final_infection:.2f} (threshold: {threshold}). "
                f"Budget remaining: {env.budget:.1f}/{env.initial_budget}. "
                f"Steps: {env._step_count}. Total reward: {env.total_reward:.2f}."
            ),
            "details": {
                "final_infection": round(final_infection, 4),
                "infection_score": round(infection_score, 4),
                "budget_score": round(budget_score, 4),
                "steps": env._step_count,
                "total_reward": round(env.total_reward, 4),
                "budget_left": round(env.budget, 2),
            },
        }

    # ── Internal helpers ─────────────────────────────────────
    def _make_obs(self, reward: float, feedback: str = "") -> MosquitoObservation:
        grid_list = self.grid.tolist() if self.grid is not None else []
        return MosquitoObservation(
            grid=grid_list,
            budget=self.budget,
            step_count=self._step_count,
            max_steps=self.max_steps,
            rainfall=self.rainfall,
            temperature=self.temperature,
            humidity=self.humidity,
            last_action=self.last_action,
            done=self._done,
            reward=reward,
            task_id=self.task_id,
            difficulty=self.difficulty,
            total_mosquito=round(self._get_total_mosquito(), 4),
            total_infection=round(self._get_total_infection(), 4),
            hotspot_count=self._get_hotspot_count(),
            feedback=feedback,
            metadata={
                "episode_id": self._episode_id,
                "task_id": self.task_id,
                "total_reward": round(self.total_reward, 4),
            },
        )

    def _get_total_infection(self) -> float:
        if self.grid is None:
            return 0.0
        return float(self.grid[:, :, 1].sum())

    def _get_total_mosquito(self) -> float:
        if self.grid is None:
            return 0.0
        return float(self.grid[:, :, 0].sum())

    def _get_hotspot_count(self) -> int:
        if self.grid is None:
            return 0
        return int(np.sum(self.grid[:, :, 0] > 0.5))

    def _update_weather(self) -> None:
        lo, hi = self.rainfall_range
        self.rainfall = float(self.rng.uniform(lo, hi))
        self.temperature = float(self.rng.uniform(20.0, 40.0))
        self.humidity = float(self.rng.uniform(0.3, 0.9))

    def _apply_action(self, a_type: str, x: int, y: int) -> None:
        if a_type == "spray":
            self.grid[x, y, 0] -= 0.4
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    self.grid[nx, ny, 0] -= 0.1
        elif a_type == "larvicide":
            self.grid[x, y, 0] -= 0.3
            self.grid[x, y, 2] -= 0.2
        elif a_type == "traps":
            self.grid[x, y, 0] -= 0.25
        elif a_type == "bed_net":
            self.grid[x, y, 1] -= 0.3
        elif a_type == "inspect":
            self.grid[x, y, 1] -= 0.05

    def _mosquito_growth(self) -> None:
        noise = self.rng.normal(0, 0.005, (GRID_SIZE, GRID_SIZE))
        growth = 0.02 * self.rainfall + 0.01 * self.humidity + noise
        self.grid[:, :, 0] += growth
        self.grid[:, :, 0] += self.grid[:, :, 2] * 0.005
        self.grid[:, :, 0] *= 0.98
        spread = np.zeros_like(self.grid[:, :, 0])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(self.grid[:, :, 0], dx, axis=0), dy, axis=1)
            spread += shifted * 0.005
        self.grid[:, :, 0] += spread

    def _infection_spread(self) -> None:
        mosquito = self.grid[:, :, 0]
        population = self.grid[:, :, 3]
        self.grid[:, :, 1] += mosquito * population * 0.01 + self.humidity * 0.002
        self.grid[:, :, 1] *= 0.97
        spread = np.zeros_like(self.grid[:, :, 1])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(self.grid[:, :, 1], dx, axis=0), dy, axis=1)
            spread += shifted * 0.003
        self.grid[:, :, 1] += spread
