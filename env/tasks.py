"""
Task Definitions — Easy, Medium, Hard

Tuned for clear difficulty differentiation:
  Easy   → score ~0.95+
  Medium → score ~0.75-0.85
  Hard   → score ~0.40-0.65
"""

TASK_EASY = {
    "name": "TaskEasy",
    "description": "Low infection rate, easy containment, high reward potential.",
    "config": {
        "max_steps": 50,
        "infection_rate": 0.05,
        "recovery_rate": 0.15,
        "death_rate": 0.003,
        "travel_rate": 0.005,
        "initial_infected": 10,
        "vaccination_rate": 0.10,
        "quarantine_effectiveness": 0.15,
    },
    "seed": 42,
    "survival_threshold": 0.90,
    "containment_bonus_threshold": 0,
    "max_acceptable_deaths": 50,
}

TASK_MEDIUM = {
    "name": "TaskMedium",
    "description": "Balanced spread with moderate difficulty. Requires strategy.",
    "config": {
        "max_steps": 50,
        "infection_rate": 0.25,
        "recovery_rate": 0.06,
        "death_rate": 0.02,
        "travel_rate": 0.04,
        "initial_infected": 80,
        "vaccination_rate": 0.03,
        "quarantine_effectiveness": 0.35,
    },
    "seed": 42,
    "survival_threshold": 0.88,
    "containment_bonus_threshold": 0,
    "max_acceptable_deaths": 150,
}

TASK_HARD = {
    "name": "TaskHard",
    "description": "High infection, fast spread, low recovery. Extremely challenging.",
    "config": {
        "max_steps": 50,
        "infection_rate": 0.60,
        "recovery_rate": 0.025,
        "death_rate": 0.05,
        "travel_rate": 0.12,
        "initial_infected": 200,
        "vaccination_rate": 0.015,
        "quarantine_effectiveness": 0.5,
    },
    "seed": 42,
    "survival_threshold": 0.85,
    "containment_bonus_threshold": 0,
    "max_acceptable_deaths": 200,
}

ALL_TASKS = [TASK_EASY, TASK_MEDIUM, TASK_HARD]


def get_task(name):
    """Get a task by name. Returns None if not found."""
    for task in ALL_TASKS:
        if task["name"] == name:
            return task
    return None
