#!/usr/bin/env python3
"""
Pandemic RL Inference — Meta PyTorch OpenEnv Hackathon
======================================================
Runs all 3 tasks (Easy/Medium/Hard), outputs strict OpenEnv log format.
Single entry point: python inference.py

Output format per task:
  [START] task=<name> env=PandemicEnv model=<agent>
  [STEP] step=<n> action=<a> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

ZERO-CRASH guarantee: every operation wrapped in try/except.
"""

import os
import sys
import time
import random

# Ensure env/ package is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Safe optional imports ────────────────────────────────────────────────────

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    pass


def clamp_reward(r):
    """Clamp reward to [0, 1] range as required by spec."""
    try:
        return round(max(0.0, min(1.0, float(r))), 2)
    except Exception:
        return 0.0


def run_task(task_def, agent, agent_name):
    """
    Run one task end-to-end, printing [START]/[STEP]/[END] lines.
    Returns (success, score).
    Never crashes — prints [END] on any error.
    """
    task_name = task_def.get("name", "Unknown")
    rewards = []

    print(f"[START] task={task_name} env=PandemicEnv model={agent_name}", flush=True)

    try:
        from env.environment import PandemicEnv
        from env.grader import grade

        env = PandemicEnv(config=task_def["config"], seed=task_def.get("seed", 42))
        obs = env.reset()
        done = False
        steps = 0
        max_steps = task_def["config"].get("max_steps", 50)
        deadline = time.time() + 5.0  # 5s safety timeout

        while not done and steps < max_steps:
            if time.time() > deadline:
                break

            # Get action from agent
            error_msg = "null"
            try:
                action = agent.act(obs)
                action = max(0, min(int(action), 6))
            except Exception as e:
                action = 0
                error_msg = str(e)

            # Step environment
            try:
                obs, reward, done, info = env.step(action)
                r = clamp_reward(reward)
                rewards.append(r)
                steps += 1
                print(
                    f"[STEP] step={steps} action={action} "
                    f"reward={r:.2f} done={'true' if done else 'false'} "
                    f"error={error_msg}",
                    flush=True,
                )
            except Exception as e:
                steps += 1
                rewards.append(0.0)
                print(
                    f"[STEP] step={steps} action={action} "
                    f"reward=0.00 done=true error={e}",
                    flush=True,
                )
                done = True

        # Grade the trajectory
        try:
            trajectory = env.get_trajectory()
            score = round(grade(trajectory, task_def), 4)
        except Exception:
            score = 0.0

        reward_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success=true steps={steps} score={score} "
            f"rewards={reward_str}",
            flush=True,
        )
        return True, score

    except Exception as e:
        # Task-level failure — still print [END]
        reward_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(
            f"[END] success=false steps={len(rewards)} score=0.0 "
            f"rewards={reward_str}",
            flush=True,
        )
        return False, 0.0


def main():
    """Run all tasks sequentially with strict output format."""

    # ── Load tasks ────────────────────────────────────────────────────────
    tasks = []
    try:
        from env.tasks import ALL_TASKS
        tasks = ALL_TASKS
    except Exception:
        # Inline fallback tasks
        tasks = [
            {
                "name": "TaskEasy",
                "config": {
                    "max_steps": 50, "infection_rate": 0.05,
                    "recovery_rate": 0.15, "death_rate": 0.003,
                    "travel_rate": 0.005, "initial_infected": 10,
                    "vaccination_rate": 0.10, "quarantine_effectiveness": 0.15,
                },
                "seed": 42, "survival_threshold": 0.90, "max_acceptable_deaths": 50,
            },
            {
                "name": "TaskMedium",
                "config": {
                    "max_steps": 50, "infection_rate": 0.25,
                    "recovery_rate": 0.06, "death_rate": 0.02,
                    "travel_rate": 0.04, "initial_infected": 80,
                    "vaccination_rate": 0.03, "quarantine_effectiveness": 0.35,
                },
                "seed": 42, "survival_threshold": 0.88, "max_acceptable_deaths": 150,
            },
            {
                "name": "TaskHard",
                "config": {
                    "max_steps": 50, "infection_rate": 0.55,
                    "recovery_rate": 0.025, "death_rate": 0.05,
                    "travel_rate": 0.10, "initial_infected": 200,
                    "vaccination_rate": 0.015, "quarantine_effectiveness": 0.5,
                },
                "seed": 42, "survival_threshold": 0.85, "max_acceptable_deaths": 200,
            },
        ]

    # ── Initialize agent ──────────────────────────────────────────────────
    agent = None
    agent_name = "RandomAgent"
    try:
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_pandemic.pt")
        try:
            from env.agents import create_agent
            agent, agent_name = create_agent(checkpoint_path=ckpt)
        except Exception:
            from env.agents import RuleBasedAgent, RandomAgent
            try:
                agent = RuleBasedAgent()
                agent_name = "RuleBasedAgent"
            except Exception:
                agent = RandomAgent()
                agent_name = "RandomAgent"
    except Exception:
        # Emergency inline agent
        class _EmergencyAgent:
            name = "EmergencyAgent"
            def act(self, obs):
                return random.randint(0, 6)
        agent = _EmergencyAgent()
        agent_name = "EmergencyAgent"

    # ── Run all tasks ─────────────────────────────────────────────────────
    for task_def in tasks:
        try:
            run_task(task_def, agent, agent_name)
        except Exception:
            # Should never reach here due to run_task's own safety,
            # but just in case — print minimal [END]
            print(
                f"[END] success=false steps=0 score=0.0 rewards=0.00",
                flush=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — nuclear crash guard, guaranteed exit(0)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # If main() itself crashes, print minimal valid output for each task
        try:
            for name in ["TaskEasy", "TaskMedium", "TaskHard"]:
                print(f"[START] task={name} env=PandemicEnv model=EmergencyFallback", flush=True)
                print(f"[STEP] step=1 action=0 reward=0.00 done=true error=critical_failure", flush=True)
                print(f"[END] success=false steps=1 score=0.0 rewards=0.00", flush=True)
        except BaseException:
            pass
    finally:
        try:
            sys.exit(0)
        except SystemExit:
            pass
