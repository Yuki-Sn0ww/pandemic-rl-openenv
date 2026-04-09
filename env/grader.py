"""
Grader — Deterministic scoring for each task
grade(trajectory, task) -> float in [0.0, 1.0]

Scoring:
  Survival  (0.0-0.5): final survival_rate vs threshold
  Containment (0.0-0.3): whether infection reached zero
  Death penalty (0.0-0.2): total deaths vs max acceptable
"""


def grade(trajectory, task):
    """
    Grade a trajectory against a task definition.

    Args:
        trajectory: list of step dicts from env.get_trajectory()
        task: task dict from tasks.py

    Returns:
        float score in [0.0, 1.0], deterministic.
    """
    try:
        if not trajectory:
            return 0.0

        final = trajectory[-1]
        info = final.get("info", {})

        survival_rate = info.get("survival_rate", 0.0)
        infected = info.get("infected", 0)
        dead = info.get("dead", 0)
        total_pop = info.get("total_population", 3000)

        # Component 1: Survival (0.0 - 0.5)
        threshold = task.get("survival_threshold", 0.80)
        if survival_rate >= threshold:
            survival_score = 0.5
        elif survival_rate >= threshold * 0.5:
            low = threshold * 0.5
            survival_score = 0.5 * (survival_rate - low) / max(threshold - low, 0.01)
        else:
            survival_score = 0.0

        # Component 2: Containment (0.0 - 0.3)
        if infected == 0:
            containment_score = 0.3
        else:
            infection_ratio = infected / max(total_pop, 1)
            containment_score = max(0.0, 0.3 * (1.0 - infection_ratio * 5.0))

        # Component 3: Death penalty (0.0 - 0.2)
        max_deaths = task.get("max_acceptable_deaths", 200)
        if dead == 0:
            death_score = 0.2
        elif dead <= max_deaths:
            death_score = 0.2 * (1.0 - dead / max(max_deaths, 1))
        else:
            death_score = 0.0

        score = survival_score + containment_score + death_score
        return round(max(0.0, min(1.0, score)), 4)

    except Exception:
        return 0.0


def grade_summary(trajectory, task):
    """Return detailed grading breakdown dict."""
    try:
        if not trajectory:
            return {"score": 0.0, "error": "empty trajectory"}

        final = trajectory[-1]
        info = final.get("info", {})
        score = grade(trajectory, task)

        return {
            "task": task.get("name", "unknown"),
            "score": score,
            "survival_rate": info.get("survival_rate", 0.0),
            "infected": info.get("infected", 0),
            "dead": info.get("dead", 0),
            "total_population": info.get("total_population", 0),
            "steps": info.get("step", 0),
            "trajectory_length": len(trajectory),
        }
    except Exception as e:
        return {"task": task.get("name", "unknown"), "score": 0.0, "error": str(e)}
