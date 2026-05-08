"""
models.py — Type-safe data contracts for the Mosquito Control Environment.

Every field is documented. Your IDE will autocomplete everything.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State


class MosquitoAction(Action):
    """
    The action an AI agent takes: choose an intervention and target cell.

    Fields:
        action_type : One of: spray, larvicide, traps, bed_net, inspect, no_op
        x           : Target grid row (0–9)
        y           : Target grid column (0–9)
    """

    action_type: str
    x: int
    y: int


class MosquitoObservation(Observation):
    """
    What the agent sees after reset() or step().

    Inherited from Observation base:
        done     : bool   — True when the episode is over.
        reward   : float  — Reward signal from the last action.
        metadata : dict   — Additional metadata.

    Custom fields:
        grid             : 10x10x4 grid — [mosquito, infection, water, population]
        budget           : Remaining budget for interventions
        step_count       : Current step number
        max_steps        : Maximum steps in this episode
        rainfall         : Current rainfall level (0–1)
        temperature      : Current temperature (20–40°C)
        humidity         : Current humidity (0–1)
        last_action      : The previous action taken (None on reset)
        task_id          : Current task name
        difficulty       : Task difficulty level
        total_mosquito   : Sum of mosquito density across all cells
        total_infection  : Sum of infection risk across all cells
        hotspot_count    : Number of cells with mosquito density > 0.5
        feedback         : Textual description of what happened this step
    """

    grid: List[List[List[float]]]
    budget: float
    step_count: int
    max_steps: int
    rainfall: float
    temperature: float
    humidity: float
    last_action: Optional[Dict[str, Any]] = None
    task_id: str = ""
    difficulty: str = ""
    total_mosquito: float = 0.0
    total_infection: float = 0.0
    hotspot_count: int = 0
    feedback: str = ""


class MosquitoState(State):
    """
    Episode metadata returned by state().

    Inherited from State base:
        episode_id : Optional[str] — Unique episode identifier.
        step_count : int           — Total steps taken this episode.

    Custom fields:
        task_id         : Which task is running.
        difficulty      : Difficulty level.
        budget          : Remaining budget.
        max_steps       : Maximum steps in this episode.
        total_reward    : Cumulative reward so far.
        total_infection : Current total infection.
        total_mosquito  : Current total mosquito density.
        done            : Whether episode is over.
    """

    task_id: str = ""
    difficulty: str = ""
    budget: float = 0.0
    max_steps: int = 50
    total_reward: float = 0.0
    total_infection: float = 0.0
    total_mosquito: float = 0.0
    done: bool = False
