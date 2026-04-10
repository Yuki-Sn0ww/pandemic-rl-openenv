import gradio as gr
from fastapi import FastAPI
from env.environment import PandemicEnv
from env.tasks import TASK_EASY
import subprocess

# ----------- API PART (IMPORTANT) ----------- #

app = FastAPI()
env = None

@app.post("/reset")
def reset():
    global env
    try:
        env = PandemicEnv(config=TASK_EASY["config"], seed=42)
        obs = env.reset()
        return {"observation": obs}
    except Exception as e:
        return {"error": f"Failed to reset environment: {e}"}

@app.post("/step")
def step(action: int):
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
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    try:
        return env.state()
    except Exception as e:
        return {"error": f"State retrieval failed: {e}"}


# ----------- UI PART ----------- #

def run_model():
    try:
        output = subprocess.check_output(["python3", "inference.py"], stderr=subprocess.STDOUT)
        return output.decode("utf-8")
    except Exception as e:
        return str(e)

demo = gr.Interface(
    fn=run_model,
    inputs=[],
    outputs="text",
    title="Pandemic RL Simulation",
    description="Click run to execute simulation"
)

# Mount UI onto FastAPI
app = gr.mount_gradio_app(app, demo, path="/")