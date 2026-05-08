"""
Interactive Gradio UI for MosquitoControl OpenEnv.
Users can see the city grid, pick actions, and watch the environment respond.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import MosquitoAction
from server.environment import MosquitoControlEnvironment, ACTION_TYPES, ACTION_COSTS, GRID_SIZE, TASKS

# ── Global state ─────────────────────────────────────────────
env = None
history = {"steps": [], "rewards": [], "infections": [], "mosquitoes": [], "budgets": []}


def make_grid_figure(grid, title="City Grid"):
    """Render 4-channel grid as a 2x2 heatmap figure."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    channel_names = ["Mosquito Density", "Infection Risk", "Water Level", "Population Density"]
    cmaps = ["YlOrRd", "Reds", "Blues", "Purples"]

    for idx, (ax, name, cmap) in enumerate(zip(axes.flat, channel_names, cmaps)):
        data = np.array([[grid[i][j][idx] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)])
        im = ax.imshow(data, vmin=0, vmax=1, cmap=cmap, aspect="equal")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                val = data[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig


def make_history_figure():
    """Render reward/infection/mosquito/budget trends."""
    if len(history["steps"]) == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No steps yet", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    fig.suptitle("Episode Trends", fontsize=12, fontweight="bold")

    axes[0].plot(history["steps"], history["rewards"], "b-o", markersize=3)
    axes[0].set_title("Step Reward")
    axes[0].set_xlabel("Step")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)

    axes[1].plot(history["steps"], history["infections"], "r-o", markersize=3)
    axes[1].set_title("Total Infection")
    axes[1].set_xlabel("Step")

    axes[2].plot(history["steps"], history["mosquitoes"], "orange", marker="o", markersize=3)
    axes[2].set_title("Total Mosquito")
    axes[2].set_xlabel("Step")

    axes[3].plot(history["steps"], history["budgets"], "g-o", markersize=3)
    axes[3].set_title("Budget Left")
    axes[3].set_xlabel("Step")

    plt.tight_layout()
    return fig


def reset_env(task_name, seed):
    """Reset the environment with chosen task and seed."""
    global env, history
    seed = int(seed)
    env = MosquitoControlEnvironment()
    obs = env.reset(seed=seed, task_id=task_name)
    history = {"steps": [], "rewards": [], "infections": [], "mosquitoes": [], "budgets": []}

    task = TASKS[task_name]
    grid_fig = make_grid_figure(obs.grid, title=f"Initial State — {task['name']} ({task_name})")
    hist_fig = make_history_figure()

    info = (
        f"## Episode Started\n"
        f"- **Task:** {task['name']} ({task_name})\n"
        f"- **Difficulty:** {task['difficulty']}\n"
        f"- **Seed:** {seed}\n"
        f"- **Budget:** {obs.budget}\n"
        f"- **Max Steps:** {obs.max_steps}\n"
        f"- **Target:** infection < {task['infection_threshold']}\n"
        f"- **Weather:** Rain={obs.rainfall:.2f}, Temp={obs.temperature:.1f}C, Humidity={obs.humidity:.2f}\n"
        f"- **Total Mosquito:** {obs.total_mosquito:.3f}\n"
        f"- **Total Infection:** {obs.total_infection:.3f}\n"
        f"- **Hotspots:** {obs.hotspot_count}\n"
    )

    return grid_fig, hist_fig, info, ""


def step_env(action_type, x, y):
    """Take one step in the environment."""
    global env, history

    if env is None:
        return None, None, "**Error:** Click Reset first!", ""
    if env._done:
        return (
            make_grid_figure(env.grid.tolist(), "Episode Over"),
            make_history_figure(),
            "**Episode is done.** Click Reset to start a new one.",
            "",
        )

    x, y = int(x), int(y)
    action = MosquitoAction(action_type=action_type, x=x, y=y)
    obs = env.step(action)

    history["steps"].append(env._step_count)
    history["rewards"].append(obs.reward)
    history["infections"].append(obs.total_infection)
    history["mosquitoes"].append(obs.total_mosquito)
    history["budgets"].append(obs.budget)

    grid_fig = make_grid_figure(obs.grid, title=f"Step {env._step_count}")
    hist_fig = make_history_figure()

    status = "DONE" if obs.done else "Running"
    info = (
        f"## Step {env._step_count} — {status}\n"
        f"- **Action:** {action_type} at ({x}, {y}) — Cost: {ACTION_COSTS[action_type]}\n"
        f"- **Reward:** {obs.reward:+.3f}\n"
        f"- **Total Reward:** {env.total_reward:.3f}\n"
        f"- **Budget Left:** {obs.budget:.1f}\n"
        f"- **Weather:** Rain={obs.rainfall:.2f}, Temp={obs.temperature:.1f}C, Humidity={obs.humidity:.2f}\n"
        f"- **Total Mosquito:** {obs.total_mosquito:.3f}\n"
        f"- **Total Infection:** {obs.total_infection:.3f}\n"
        f"- **Hotspots:** {obs.hotspot_count}\n"
    )
    if obs.done:
        info += f"\n### Episode Complete!\nFinal reward: **{env.total_reward:.3f}**\n"

    log_line = (
        f"Step {env._step_count}: {action_type}({x},{y}) → "
        f"reward={obs.reward:+.3f}  budget={obs.budget:.1f}  "
        f"infection={obs.total_infection:.2f}  mosquito={obs.total_mosquito:.2f}  "
        f"hotspots={obs.hotspot_count}"
    )

    return grid_fig, hist_fig, info, log_line


def auto_run(task_name, seed, num_steps):
    """Run N random steps automatically."""
    global env, history
    seed = int(seed)
    num_steps = int(num_steps)

    env = MosquitoControlEnvironment()
    obs = env.reset(seed=seed, task_id=task_name)
    history = {"steps": [], "rewards": [], "infections": [], "mosquitoes": [], "budgets": []}

    log_lines = []
    for i in range(num_steps):
        a_type = env.rng.choice(ACTION_TYPES)
        action = MosquitoAction(
            action_type=a_type,
            x=int(env.rng.randint(0, GRID_SIZE)),
            y=int(env.rng.randint(0, GRID_SIZE)),
        )
        obs = env.step(action)
        history["steps"].append(env._step_count)
        history["rewards"].append(obs.reward)
        history["infections"].append(obs.total_infection)
        history["mosquitoes"].append(obs.total_mosquito)
        history["budgets"].append(obs.budget)
        log_lines.append(
            f"Step {env._step_count}: {action.action_type}({action.x},{action.y}) → "
            f"reward={obs.reward:+.3f}  budget={obs.budget:.1f}  "
            f"inf={obs.total_infection:.2f}  mosq={obs.total_mosquito:.2f}"
        )
        if obs.done:
            log_lines.append(f"--- Episode ended at step {env._step_count} ---")
            break

    grid_fig = make_grid_figure(obs.grid, title=f"After {env._step_count} Steps")
    hist_fig = make_history_figure()
    info = (
        f"## Auto-Run Complete\n"
        f"- **Task:** {task_name} | **Steps:** {env._step_count}\n"
        f"- **Total Reward:** {env.total_reward:.3f}\n"
        f"- **Budget Left:** {env.budget:.1f}\n"
        f"- **Final Infection:** {env._get_total_infection():.3f}\n"
        f"- **Final Mosquito:** {env._get_total_mosquito():.3f}\n"
        f"- **Hotspots:** {env._get_hotspot_count()}\n"
        f"- **Done:** {env._done}\n"
    )

    return grid_fig, hist_fig, info, "\n".join(log_lines)


def run_grader(task_name):
    """Run the grader for a specific task."""
    result = MosquitoControlEnvironment.run_grader(task_name)
    passed = "PASS" if result["passed"] else "FAIL"
    info = (
        f"## Grader Result: {task_name} — {passed}\n"
        f"- **Score:** {result['score']:.4f} / 1.0\n"
        f"- **Feedback:** {result['feedback']}\n"
    )
    if "details" in result:
        d = result["details"]
        info += (
            f"- **Infection Score:** {d.get('infection_score', 0):.4f}\n"
            f"- **Budget Score:** {d.get('budget_score', 0):.4f}\n"
            f"- **Steps:** {d.get('steps', 0)}\n"
            f"- **Total Reward:** {d.get('total_reward', 0):.4f}\n"
        )
    return info


def run_all_graders():
    """Run graders for all tasks."""
    lines = ["## All Task Results\n"]
    total = 0
    for tid in TASKS:
        result = MosquitoControlEnvironment.run_grader(tid)
        passed = "PASS" if result["passed"] else "FAIL"
        lines.append(f"**{tid}** ({TASKS[tid]['difficulty']}): score={result['score']:.4f} {passed}")
        total += result["score"]
    avg = total / len(TASKS)
    lines.append(f"\n**Average Score: {avg:.4f}**")
    return "\n\n".join(lines)


# ── Build UI ─────────────────────────────────────────────────
with gr.Blocks(title="MosquitoControl OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# MosquitoControl OpenEnv\n"
        "Adaptive city-wide mosquito suppression RL environment. "
        "Control mosquito outbreaks on a 10x10 city grid under budget and weather constraints.\n\n"
        "| Action | Cost | Effect |\n"
        "|--------|------|--------|\n"
        "| spray | $10 | -0.4 mosquito at target + neighbors |\n"
        "| larvicide | $7 | -0.3 mosquito, -0.2 water |\n"
        "| traps | $6 | -0.25 mosquito |\n"
        "| bed_net | $3 | -0.3 infection |\n"
        "| inspect | $2 | -0.05 infection |\n"
        "| no_op | $0 | Nothing |"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            task_dd = gr.Dropdown(
                choices=list(TASKS.keys()),
                value="easy",
                label="Task",
            )
            seed_num = gr.Number(value=42, label="Seed", precision=0)
            reset_btn = gr.Button("Reset Environment", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### Manual Step")
            action_dd = gr.Dropdown(choices=ACTION_TYPES, value="spray", label="Action")
            x_slider = gr.Slider(0, 9, value=0, step=1, label="Target Row (x)")
            y_slider = gr.Slider(0, 9, value=0, step=1, label="Target Col (y)")
            step_btn = gr.Button("Take Step", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### Auto-Run")
            num_steps = gr.Slider(5, 60, value=20, step=1, label="Number of Steps")
            auto_btn = gr.Button("Auto-Run (Random Agent)", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### Grader")
            grade_btn = gr.Button("Run Grader (selected task)", variant="secondary")
            grade_all_btn = gr.Button("Run All Graders", variant="secondary")
            grader_output = gr.Markdown("")

        with gr.Column(scale=3):
            info_md = gr.Markdown("Click **Reset Environment** to begin.")
            grid_plot = gr.Plot(label="City Grid")
            hist_plot = gr.Plot(label="Trends")
            log_box = gr.Textbox(label="Step Log", lines=10, interactive=False)

    # Wire events
    reset_btn.click(
        fn=reset_env,
        inputs=[task_dd, seed_num],
        outputs=[grid_plot, hist_plot, info_md, log_box],
    )
    step_btn.click(
        fn=step_env,
        inputs=[action_dd, x_slider, y_slider],
        outputs=[grid_plot, hist_plot, info_md, log_box],
    )
    auto_btn.click(
        fn=auto_run,
        inputs=[task_dd, seed_num, num_steps],
        outputs=[grid_plot, hist_plot, info_md, log_box],
    )
    grade_btn.click(fn=run_grader, inputs=[task_dd], outputs=[grader_output])
    grade_all_btn.click(fn=run_all_graders, inputs=[], outputs=[grader_output])


if __name__ == "__main__":
    demo.launch(server_port=7861)
