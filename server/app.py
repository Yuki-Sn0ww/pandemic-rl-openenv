"""
Pandemic RL — OpenEnv API Server
=================================
FastAPI server exposing the Pandemic RL environment for evaluation.

Endpoints:
  POST /reset  → Initialize/reset environment, returns observation
  POST /step   → Take an action, returns (observation, reward, done, info)
  GET  /state  → Returns full environment state
"""

import sys
import os

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI

app = FastAPI(
    title="Pandemic RL",
    description="OpenEnv Pandemic RL Simulation API",
    version="1.0.0",
)

# Global environment instance
env = None


@app.get("/")
def root():
    """Health check."""
    return {"message": "Pandemic RL Server Running", "status": "ok"}


@app.post("/reset")
def reset():
    """Initialize or reset the environment. Must be called before /step or /state."""
    global env
    try:
        from env.environment import PandemicEnv
        from env.tasks import TASK_EASY

        env = PandemicEnv(config=TASK_EASY["config"], seed=42)
        obs = env.reset()
        return {"observation": obs}
    except Exception as e:
        return {"error": f"Failed to reset environment: {e}"}


@app.post("/step")
def step(action: int = 0):
    """Take one step in the environment."""
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        return {"error": f"Step failed: {e}"}


@app.get("/state")
def state():
    """Return full environment state."""
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    try:
        return env.state()
    except Exception as e:
        return {"error": f"State retrieval failed: {e}"}


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    """Required by pyproject.toml scripts entry."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)