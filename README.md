# 🦠 Pandemic RL — Meta PyTorch OpenEnv Hackathon

A crash-proof reinforcement learning system that models pandemic containment across 3 cities with 3 difficulty-tiered tasks.

## ▶️ How to Run

```bash
pip install -r requirements.txt
python inference.py
```

**Docker:**
```bash
docker build -t pandemic-rl .
docker run pandemic-rl
```

## 🌍 Environment Description

A 3-city SIR (Susceptible-Infected-Recovered) pandemic model. Each city has a population of 1,000. Disease spreads within cities and between cities via travel. The agent intervenes through quarantine and vaccination policies.

### Observation Space

**Type:** `Box(12)` — 12 continuous floats in `[0.0, 1.0]`

| Index | Meaning                          |
|-------|----------------------------------|
| 0-3   | City 0: susceptible, infected, recovered, dead (÷1000) |
| 4-7   | City 1: susceptible, infected, recovered, dead (÷1000) |
| 8-11  | City 2: susceptible, infected, recovered, dead (÷1000) |

### Action Space

**Type:** `Discrete(7)`

| Action | Effect               |
|--------|----------------------|
| 0      | Do nothing           |
| 1      | Quarantine City 0    |
| 2      | Quarantine City 1    |
| 3      | Quarantine City 2    |
| 4      | Vaccinate City 0     |
| 5      | Vaccinate City 1     |
| 6      | Vaccinate City 2     |

### OpenEnv API

```python
from env.environment import PandemicEnv

env = PandemicEnv(config={...}, seed=42)
obs = env.reset()
obs, reward, done, info = env.step(action)
state = env.state()
trajectory = env.get_trajectory()
```

## 📋 Tasks

Three difficulty tiers using the same environment with different parameters:

| Task | Infection Rate | Recovery | Death Rate | Initial Infected | Travel |
|------|---------------|----------|------------|------------------|--------|
| **TaskEasy** | 0.05 | 0.15 | 0.003 | 10 | 0.005 |
| **TaskMedium** | 0.25 | 0.06 | 0.02 | 80 | 0.04 |
| **TaskHard** | 0.55 | 0.025 | 0.04 | 200 | 0.10 |

## 📊 Scoring Method

`grade(trajectory, task) → float [0.0, 1.0]`

| Component | Weight | Criteria |
|-----------|--------|----------|
| **Survival** | 50% | Final survival rate vs task threshold |
| **Containment** | 30% | Whether infections reached zero |
| **Death Penalty** | 20% | Total deaths vs maximum acceptable |

Scoring is **deterministic** and **reproducible** (fixed seed=42).

## 🛡️ Fallback Safety

```
PPO Agent (torch + checkpoint)
  └─ fails → RuleBased Agent (heuristic)
       └─ fails → Random Agent (stdlib only)
            └─ fails → Emergency inline agent
```

Every operation is `try/except` wrapped. Nuclear fallback prints valid `START...END` output even if `main()` itself crashes.

**Key guarantees:**
- **Deterministic evaluation** (seed=42) — identical output on every run
- **Crash-proof fallback system** — zero-crash guarantee under any condition

## 🌐 Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `API_BASE_URL` | LLM API endpoint | No |
| `MODEL_NAME` | Model name | No |
| `HF_TOKEN` | HF auth token | No |
| `LOCAL_IMAGE_NAME` | Docker image | No |

All optional. System works without any.

## 📁 Project Structure

```
├── inference.py          # Entry point — runs all 3 tasks
├── openenv.yaml          # OpenEnv specification
├── requirements.txt      # Dependencies
├── Dockerfile            # Container support
├── README.md             # This file
└── env/
    ├── __init__.py
    ├── environment.py    # PandemicEnv (reset/step/state)
    ├── tasks.py          # TaskEasy, TaskMedium, TaskHard
    ├── grader.py         # grade(trajectory, task) → 0.0-1.0
    └── agents.py         # PPO, RuleBased, Random + fallback factory
```

## 📜 License

MIT — Meta PyTorch OpenEnv Hackathon 2026
