from fastapi import FastAPI
from env.environment import PandemicEnv

app = FastAPI()

env = None

@app.post("/reset")
def reset():
    global env
    env = PandemicEnv(config={}, seed=42)
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
def step(action: int):
    global env
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }