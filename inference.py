"""
inference.py — LLM Baseline Agent for the Mosquito Control Environment.

Usage:
    set API_BASE_URL=https://api.groq.com/openai/v1
    set MODEL_NAME=llama-3.1-8b-instant
    set HF_TOKEN=your_key
    python inference.py --url http://localhost:7860
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting baseline inference script...")

try:
    from openai import OpenAI
    print("OpenAI package loaded OK")
except ImportError:
    print("ERROR: openai not installed. Run: pip install openai")
    sys.exit(1)

try:
    from client import MosquitoEnv
    from models import MosquitoAction
    print("Client and models loaded OK")
except ImportError as e:
    print(f"ERROR importing client/models: {e}")
    sys.exit(1)


# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert public health agent managing mosquito control across a 10x10 city grid.

Each cell has: mosquito_density (0-1), infection_risk (0-1), water_level (0-1), population_density (0-1).

Available actions and costs:
- spray (cost 10): reduces mosquito by 0.4 at target + 0.1 at neighbors
- larvicide (cost 7): reduces mosquito by 0.3 and water by 0.2
- traps (cost 6): reduces mosquito by 0.25
- bed_net (cost 3): reduces infection by 0.3
- inspect (cost 2): reduces infection by 0.05
- no_op (cost 0): do nothing

Strategy tips:
- Target high-mosquito cells first (hotspots)
- Use spray on clusters, larvicide on high-water cells
- Use bed_net on high-infection + high-population cells
- Conserve budget for later steps
- Weather affects breeding: high rain = more mosquitoes

Return ONLY a JSON object: {"action_type": "...", "x": N, "y": N}
No explanation, no markdown. Just raw JSON."""


def build_prompt(obs) -> str:
    """Build user message from observation."""
    parts = [
        f"Step {obs.step_count}/{obs.max_steps} | Budget: {obs.budget:.1f} | "
        f"Infection: {obs.total_infection:.2f} | Mosquito: {obs.total_mosquito:.2f} | "
        f"Hotspots: {obs.hotspot_count}",
        f"Weather: rain={obs.rainfall:.2f}, temp={obs.temperature:.1f}C, humidity={obs.humidity:.2f}",
    ]

    if obs.feedback:
        parts.append(f"Last result: {obs.feedback[:200]}")

    # Find top hotspot cells
    grid = obs.grid
    cells = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            cells.append((grid[i][j][0], grid[i][j][1], grid[i][j][2], i, j))
    cells.sort(reverse=True)

    parts.append("\nTop-5 hotspot cells (mosquito, infection, water, row, col):")
    for m, inf, w, x, y in cells[:5]:
        parts.append(f"  ({x},{y}): mosq={m:.3f} inf={inf:.3f} water={w:.3f}")

    parts.append(f"\nChoose an action. Budget left: {obs.budget:.1f}")
    return "\n".join(parts)


def run_episode(env, llm_client, model_name, task_id, verbose=True):
    """Run one full episode. Returns final score 0.0-1.0."""
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Task: {task_id.upper()}")
        print(f"{'─' * 60}")

    result = env.reset(task_id=task_id)
    obs = result.observation

    print(f"[START] {json.dumps({'task_id': task_id, 'difficulty': obs.difficulty, 'budget': obs.budget, 'max_steps': obs.max_steps})}")

    if verbose:
        print(f"  Budget: {obs.budget}, Max steps: {obs.max_steps}")
        print(f"  Initial infection: {obs.total_infection:.2f}, mosquito: {obs.total_mosquito:.2f}")

    step = 0
    while not result.done:
        prompt = build_prompt(obs)

        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=100,
            )
            text = response.choices[0].message.content.strip()
            if "{" in text:
                text = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(text)
            action = MosquitoAction(
                action_type=data.get("action_type", "no_op"),
                x=int(data.get("x", 0)),
                y=int(data.get("y", 0)),
            )
        except Exception as e:
            if verbose:
                print(f"  LLM error: {e}, using no_op")
            action = MosquitoAction(action_type="no_op", x=0, y=0)

        result = env.step(action)
        obs = result.observation
        step += 1

        print(f"[STEP] {json.dumps({'task_id': task_id, 'step': step, 'action': {'action_type': action.action_type, 'x': action.x, 'y': action.y}, 'reward': round(result.reward or 0, 4), 'done': result.done, 'budget': round(obs.budget, 1), 'infection': round(obs.total_infection, 4), 'mosquito': round(obs.total_mosquito, 4)})}")

        if verbose:
            print(f"  Step {step}: {action.action_type}({action.x},{action.y}) → "
                  f"reward={result.reward or 0:+.3f} budget={obs.budget:.1f} "
                  f"inf={obs.total_infection:.2f} mosq={obs.total_mosquito:.2f}")

    final_score = max(0.0, min(1.0, (result.reward or 0)))
    print(f"[END] {json.dumps({'task_id': task_id, 'score': round(final_score, 4), 'steps': step, 'total_infection': round(obs.total_infection, 4), 'budget_left': round(obs.budget, 1)})}")

    return final_score


def main():
    parser = argparse.ArgumentParser(description="Mosquito Control LLM Baseline")
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    api_base_url = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    model_name = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

    if not hf_token:
        print("\nERROR: No API key found!")
        print("Set one of: HF_TOKEN, OPENAI_API_KEY")
        sys.exit(1)

    print(f"API key found: {hf_token[:8]}...")

    llm_client = OpenAI(api_key=hf_token, base_url=api_base_url)

    task_ids = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    verbose = not args.quiet

    print(f"\n{'=' * 60}")
    print(f"Mosquito Control Environment — LLM Baseline ({model_name})")
    print(f"Server: {args.url}")
    print(f"Tasks: {', '.join(task_ids)}")
    print(f"{'=' * 60}")

    scores = {}
    with MosquitoEnv(base_url=args.url).sync() as env:
        for task_id in task_ids:
            try:
                score = run_episode(env, llm_client, model_name, task_id, verbose=verbose)
                scores[task_id] = score
            except Exception as exc:
                print(f"ERROR on task {task_id}: {exc}")
                import traceback
                traceback.print_exc()
                scores[task_id] = 0.0

    print(f"\n{'=' * 60}")
    print(f"BASELINE RESULTS")
    print(f"{'=' * 60}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        status = "PASS" if score >= 0.6 else "FAIL"
        print(f"  {task_id:8s} [{bar:<20}] {score:.4f} {status}")

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'─' * 45}")
    print(f"  Average: {avg:.4f}")
    print(f"{'=' * 60}\n")

    print(json.dumps({
        "model": model_name,
        "scores": scores,
        "average_score": round(avg, 4),
    }, indent=2))

    return scores


if __name__ == "__main__":
    main()
