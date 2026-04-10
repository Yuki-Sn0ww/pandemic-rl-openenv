#!/usr/bin/env python3
"""
Pandemic RL Inference — Meta PyTorch OpenEnv Hackathon
======================================================
Runs all 3 tasks (Easy/Medium/Hard), outputs strict OpenEnv log format.
Single entry point: python inference.py

Uses OpenAI client for LLM-based decision making.
Falls back to rule-based heuristic if LLM fails.

Output format per task:
  [START] task=<name> env=PandemicEnv model=<agent>
  [STEP] step=<n> action=<a> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from pandemic_rl.client import PandemicEnvClient
from pandemic_rl.models import PandemicAction
from env.tasks import ALL_TASKS
from env.grader import grade

# ─── Environment Variables ───
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("BENCHMARK", "pandemic_rl")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # If using docker image
TEMPERATURE = 0.7
MAX_TOKENS = 50

# ─── LLM System Prompt ───
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a CDC health director managing a pandemic.
    The simulation models 3 cities with Susceptible-Infected-Recovered-Dead populations.
    Available actions:
    0 = Do nothing
    1, 2, 3 = Quarantine City 0, 1, or 2 (reduces infection spread but causes economic damage)
    4, 5, 6 = Vaccinate City 0, 1, or 2 (moves susceptible to recovered directly)

    Reply with exactly one integer (0-6) representing your chosen action.
    """
).strip()


# ─── Logging Functions (OpenEnv strict format) ───

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── Helper Functions ───

def build_user_prompt(step: int, obs, last_reward: float) -> str:
    """Build the prompt sent to the LLM with current observation data."""
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation (Susceptible/Infected/Recovered/Dead):
        City 0: {obs.city_data[0]:.3f} / {obs.city_data[1]:.3f} / {obs.city_data[2]:.3f} / {obs.city_data[3]:.3f}
        City 1: {obs.city_data[4]:.3f} / {obs.city_data[5]:.3f} / {obs.city_data[6]:.3f} / {obs.city_data[7]:.3f}
        City 2: {obs.city_data[8]:.3f} / {obs.city_data[9]:.3f} / {obs.city_data[10]:.3f} / {obs.city_data[11]:.3f}

        Total Stats:
        Susceptible: {obs.susceptible}
        Infected: {obs.infected}
        Recovered: {obs.recovered}
        Dead: {obs.dead}
        Survival Rate: {obs.survival_rate:.2f}

        Last reward: {last_reward:.2f}
        Choose next action (0-6):
        """
    ).strip()


def clamp_reward(r) -> float:
    """Clamp reward to [0.0, 1.0] range."""
    try:
        return max(0.0, min(1.0, float(r)))
    except Exception:
        return 0.0


def get_model_action(client: OpenAI, step: int, obs, last_reward: float) -> str:
    """
    Query the LLM for an action decision.
    Falls back to a rule-based heuristic if the LLM call fails.
    """
    user_prompt = build_user_prompt(step, obs, last_reward)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Extract first valid digit (0-6) from response
        for char in text:
            if char.isdigit():
                action = int(char)
                if 0 <= action <= 6:
                    return str(action)
        return "0"
    except Exception:
        # Fallback: rule-based heuristic — quarantine most infected city
        if obs.city_data[1] > 0.05:
            return "1"
        if obs.city_data[5] > 0.05:
            return "2"
        if obs.city_data[9] > 0.05:
            return "3"
        return "4"


# ─── Task Runner ───

async def run_task(task_def: dict, env_client: PandemicEnvClient, client: OpenAI) -> None:
    """Run a single task: reset env, loop steps, grade, log results."""
    task_name = task_def["name"]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    trajectory = []

    try:
        # Reset environment for this task
        result = await env_client.reset(task_name=task_name)
        obs = result.observation
        last_reward = 0.0

        trajectory.append({
            "step": 0,
            "observation": obs.city_data,
            "action": None,
            "reward": 0.0,
            "info": {
                "susceptible": obs.susceptible,
                "infected": obs.infected,
                "recovered": obs.recovered,
                "dead": obs.dead,
                "survival_rate": obs.survival_rate,
                "step": obs.step,
            },
        })

        # Step loop (max 50 steps)
        for step in range(1, 51):
            if obs.done:
                break

            # Get action from LLM (or fallback)
            action_str = get_model_action(client, step, obs, last_reward)
            try:
                action_int = int(action_str)
            except ValueError:
                action_int = 0

            action = PandemicAction(action=action_int)

            # Execute step
            error = None
            try:
                result = await env_client.step(action)
                obs = result.observation
                reward = clamp_reward(result.reward or 0.0)
                done = obs.done
            except Exception as e:
                error = str(e)
                done = True
                reward = 0.0

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            trajectory.append({
                "step": step,
                "observation": obs.city_data,
                "action": action_int,
                "reward": reward,
                "info": {
                    "susceptible": obs.susceptible,
                    "infected": obs.infected,
                    "recovered": obs.recovered,
                    "dead": obs.dead,
                    "survival_rate": obs.survival_rate,
                    "step": obs.step,
                },
            })

            if done:
                break

        # Grade using full trajectory
        score = grade(trajectory, task_def)
        score = min(max(score, 0.0), 1.0)
        success = score >= task_def.get("survival_threshold", 0.1)

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ─── Main Entry Point ───

async def main() -> None:
    """Run all tasks sequentially against the OpenEnv server."""
    # Initialize OpenAI client with HF-compatible API
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "DUMMY")

    # Connect to the environment server via WebSocket
    try:
        if IMAGE_NAME:
            env_client = await PandemicEnvClient.from_docker_image(IMAGE_NAME)
            try:
                for task in ALL_TASKS:
                    await run_task(task, env_client, client)
            finally:
                await env_client.close()
        else:
            async with PandemicEnvClient(base_url="http://localhost:8000") as env:
                for task in ALL_TASKS:
                    await run_task(task, env, client)
    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
