"""
client.py — HTTP client for the Mosquito Control Environment.

Usage:
    from client import MosquitoEnv
    from models import MosquitoAction

    with MosquitoEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id="easy")
        while not result.done:
            action = MosquitoAction(action_type="spray", x=3, y=3)
            result = env.step(action)
            print(f"Reward: {result.reward}, Done: {result.done}")
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from models import MosquitoAction, MosquitoObservation, MosquitoState


class MosquitoEnv(EnvClient[MosquitoAction, MosquitoObservation, MosquitoState]):
    """Type-safe client for the Mosquito Control environment."""

    def _step_payload(self, action: MosquitoAction) -> Dict[str, Any]:
        """Serialize action to wire format."""
        return {
            "action_type": action.action_type,
            "x": action.x,
            "y": action.y,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MosquitoObservation]:
        """Deserialize server response to StepResult."""
        obs_data = payload.get("observation", payload)
        done = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward"))

        obs = MosquitoObservation(
            grid=obs_data.get("grid", []),
            budget=obs_data.get("budget", 0),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 50),
            rainfall=obs_data.get("rainfall", 0),
            temperature=obs_data.get("temperature", 0),
            humidity=obs_data.get("humidity", 0),
            last_action=obs_data.get("last_action"),
            done=done,
            reward=reward,
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            total_mosquito=obs_data.get("total_mosquito", 0),
            total_infection=obs_data.get("total_infection", 0),
            hotspot_count=obs_data.get("hotspot_count", 0),
            feedback=obs_data.get("feedback", ""),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> MosquitoState:
        """Deserialize state payload."""
        return MosquitoState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", ""),
            budget=payload.get("budget", 0),
            max_steps=payload.get("max_steps", 50),
            total_reward=payload.get("total_reward", 0),
            total_infection=payload.get("total_infection", 0),
            total_mosquito=payload.get("total_mosquito", 0),
            done=payload.get("done", False),
        )
