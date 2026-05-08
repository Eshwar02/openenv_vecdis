"""
server/app.py — FastAPI server for the Mosquito Control Environment.

Auto-creates /reset /step /state /health /ws /docs via create_fastapi_app.
Custom endpoints: /tasks, /grader, /baseline
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app
from models import MosquitoAction, MosquitoObservation
from server.environment import MosquitoControlEnvironment, TASKS

# Singleton factory — HTTP endpoints are stateless by default in OpenEnv,
# so we share a single env instance across all HTTP requests.
_shared_env = None

def _env_factory():
    global _shared_env
    if _shared_env is None:
        _shared_env = MosquitoControlEnvironment()
    return _shared_env

# Auto-creates /reset /step /state /health /ws /docs
app = create_fastapi_app(_env_factory, MosquitoAction, MosquitoObservation)


@app.get("/")
def root():
    """Root endpoint with environment info."""
    return JSONResponse(content={
        "name": "Mosquito Control Environment",
        "version": "1.0.0",
        "status": "running",
        "description": "OpenEnv environment for adaptive city-wide mosquito suppression",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "reset": "/reset (POST)",
            "step": "/step (POST)",
            "state": "/state (GET)",
            "tasks": "/tasks",
            "grader": "/grader",
            "baseline": "/baseline",
        },
    })


@app.get("/tasks", tags=["Competition"])
def get_tasks():
    """List all available tasks with their descriptions and action schema."""
    return JSONResponse(content={
        "tasks": MosquitoControlEnvironment.list_tasks(),
        "total": len(TASKS),
        "action_schema": {
            "action_type": "string — one of: spray, larvicide, traps, bed_net, inspect, no_op",
            "x": "int — target grid row (0–9)",
            "y": "int — target grid column (0–9)",
        },
        "action_costs": {
            "spray": 10, "larvicide": 7, "traps": 6,
            "bed_net": 3, "inspect": 2, "no_op": 0,
        },
    })


@app.post("/grader", tags=["Competition"])
def run_grader(task_id: str = "easy"):
    """Run the grader for a specific task using a random baseline agent."""
    result = MosquitoControlEnvironment.run_grader(task_id)
    return JSONResponse(content=result)


@app.get("/baseline", tags=["Competition"])
def run_baseline():
    """Run all task graders and return baseline scores."""
    baseline_scores = {}
    for task_id in TASKS:
        result = MosquitoControlEnvironment.run_grader(task_id)
        baseline_scores[task_id] = {
            "score": result["score"],
            "passed": result["passed"],
            "feedback": result["feedback"],
        }
    avg = sum(v["score"] for v in baseline_scores.values()) / len(baseline_scores)
    return JSONResponse(content={
        "baseline_agent": "random (uniform random actions)",
        "results": baseline_scores,
        "average_score": round(avg, 4),
    })


def main():
    """Entry point for direct execution."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
