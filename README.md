---
title: Mosquito Control Environment
emoji: 🦟
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# MosquitoControlEnv: Adaptive City-Wide Mosquito Suppression

A fully self-contained OpenEnv-compatible reinforcement learning environment where an AI agent learns to control mosquito outbreaks across a dynamic 10×10 city grid under budget, weather, and hotspot constraints.

## Problem Statement

Mosquito-borne diseases (dengue, malaria, chikungunya, Zika) are heavily influenced by standing water, rainfall, urban density, and delayed interventions. Health departments cannot intervene everywhere at once.

This environment simulates the real decision problem: **where should limited mosquito-control resources be deployed at every time step to minimize future outbreak risk?**

This is a **resource allocation + delayed reward** RL problem with real-world utility for public health planning.

## Environment Design

### Grid World
- **10×10 city grid** — each cell represents a city block
- **4 channels per cell**: mosquito density, infection risk, water level, population density (all 0–1)
- Seeded hotspots create initial outbreak clusters

### Actions
| Action | Cost | Effect |
|--------|------|--------|
| `spray` | $10 | -0.4 mosquito at target + -0.1 neighbors |
| `larvicide` | $7 | -0.3 mosquito, -0.2 water |
| `traps` | $6 | -0.25 mosquito |
| `bed_net` | $3 | -0.3 infection |
| `inspect` | $2 | -0.05 infection (awareness) |
| `no_op` | $0 | No intervention |

### Dynamics (per step)
1. **Weather updates**: rainfall, temperature, humidity change each step
2. **Mosquito growth**: influenced by rainfall, humidity, water levels, spatial spread from neighbors
3. **Natural decay**: mosquitoes and infections decay slightly each step
4. **Infection spread**: driven by mosquito × population density + neighbor diffusion
5. **Action effects**: immediate intervention on target cell

### Reward Function
```
reward = -infection_sum × 0.5
         -mosquito_sum × 0.3
         -action_cost × 0.1
         +infection_reduction × 8.0
         +hotspot_control_bonus
```
Clipped to [-50, 50]. Invalid actions: -5 penalty. Over-budget: -3 penalty.

## Tasks & Graders

All graders return deterministic scores in [0.0, 1.0].
Scoring: 70% infection control + 30% budget efficiency.

| Task | Difficulty | Objective | Budget | Steps |
|------|-----------|-----------|--------|-------|
| `easy` | Easy | Reduce infection below 5.0 | 100 | 50 |
| `medium` | Medium | Control with half budget (infection < 10.0) | 50 | 50 |
| `hard` | Hard | Survive monsoon (infection < 15.0, high rainfall) | 120 | 60 |

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Start Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Test Endpoints
```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl http://localhost:7860/baseline
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type": "spray", "x": 3, "y": 3}'
curl http://localhost:7860/state
```

### Interactive UI
```bash
python app_ui.py
# Opens at http://localhost:7861
```

### Run Inference (requires LLM API key)
```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export HF_TOKEN=your-groq-key
python inference.py --url http://localhost:7860
```

### Docker
```bash
docker build -t mosquito-control-env .
docker run -p 7860:7860 mosquito-control-env
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start new episode (accepts `task_id`, `seed`) |
| POST | `/step` | Execute action (`action_type`, `x`, `y`) |
| GET | `/state` | Current episode metadata |
| GET | `/tasks` | List tasks with action schema |
| POST | `/grader` | Grade a task (random baseline) |
| GET | `/baseline` | Run all graders, return scores |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger API docs |

## Project Structure
```
mosquito_control_env/
├── models.py          # Pydantic models (MosquitoAction, MosquitoObservation, MosquitoState)
├── client.py          # EnvClient for remote interaction
├── inference.py       # LLM baseline agent with [START]/[STEP]/[END] logs
├── tasks.py           # Standalone task runner (optional)
├── app_ui.py          # Interactive Gradio UI
├── openenv.yaml       # OpenEnv manifest
├── Dockerfile         # HF Spaces deployment
├── requirements.txt   # Dependencies
├── README.md
├── server/
│   ├── __init__.py
│   ├── app.py         # FastAPI server (create_fastapi_app)
│   └── environment.py # Core simulation logic
└── demo/
    └── demo.py        # Terminal demo
```

## Architecture

```
┌──────────────┐     HTTP/WS      ┌─────────────────────┐
│  LLM Agent   │◄────────────────►│  FastAPI Server      │
│  (inference)  │  /reset /step    │  (server/app.py)     │
│              │  /state           │                     │
└──────────────┘                  │  ┌─────────────────┐ │
                                  │  │ Environment      │ │
┌──────────────┐                  │  │ - 10x10 grid    │ │
│  Gradio UI   │◄────direct───────│  │ - weather sim   │ │
│  (app_ui.py) │                  │  │ - budget mgmt   │ │
└──────────────┘                  │  │ - graders       │ │
                                  │  └─────────────────┘ │
                                  └─────────────────────┘
```

This project implements a fully self-contained OpenEnv-compatible reinforcement learning environment for adaptive mosquito control. It is designed to run offline, includes task definitions, reward logic, and a reproducible demo.
