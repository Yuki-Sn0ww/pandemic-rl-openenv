"""
OpenEnv Environment Wrapper for PandemicEnv
============================================
Wraps the core env/environment.py PandemicEnv in OpenEnv's Environment base class.
The Environment class is Generic[ActT, ObsT, StateT] and requires:
  - reset() -> ObsT
  - step(action: ActT) -> ObsT
  - state (property) -> StateT
"""

from typing import Any, Optional

from pydantic import Field
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env.environment import PandemicEnv
from env.tasks import get_task, TASK_EASY
from pandemic_rl.models import PandemicAction, PandemicObservation


class PandemicState(State):
    """Full environment state for OpenEnv compliance."""
    cities: list = Field(default_factory=list)
    step_num: int = 0
    done: bool = False
    quarantined: list = Field(default_factory=list)
    vaccinated: list = Field(default_factory=list)
    susceptible: int = 0
    infected: int = 0
    recovered: int = 0
    dead: int = 0
    total_population: int = 3000
    survival_rate: float = 1.0


class PandemicEnvironment(Environment[PandemicAction, PandemicObservation, PandemicState]):
    """OpenEnv-compliant wrapper around PandemicEnv."""

    def __init__(self):
        super().__init__()
        self.env = None
        self.current_task = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> PandemicObservation:
        """Reset environment for a given task. Returns typed observation."""
        self._reset_rubric()

        task = get_task(task_name) if task_name else TASK_EASY
        if task is None:
            task = TASK_EASY

        self.current_task = task
        config = task.get("config", {})
        env_seed = seed if seed is not None else task.get("seed", 42)

        self.env = PandemicEnv(config=config, seed=env_seed)
        obs = self.env.reset()
        info = self.env._info()

        return PandemicObservation(
            city_data=obs,
            susceptible=info["susceptible"],
            infected=info["infected"],
            recovered=info["recovered"],
            dead=info["dead"],
            survival_rate=info["survival_rate"],
            step=info["step"],
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: PandemicAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PandemicObservation:
        """Execute one step. Returns observation (OpenEnv style)."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        obs, reward, done, info = self.env.step(action.action)

        observation = PandemicObservation(
            city_data=obs,
            susceptible=info["susceptible"],
            infected=info["infected"],
            recovered=info["recovered"],
            dead=info["dead"],
            survival_rate=info["survival_rate"],
            step=info["step"],
            done=done,
            reward=reward,
        )

        return observation

    @property
    def state(self) -> PandemicState:
        """Return full environment state (OpenEnv spec)."""
        if self.env is None:
            return PandemicState()

        raw = self.env.state()
        return PandemicState(
            cities=raw.get("cities", []),
            step_num=raw.get("step", 0),
            done=raw.get("done", False),
            quarantined=raw.get("quarantined", []),
            vaccinated=raw.get("vaccinated", []),
            susceptible=raw.get("susceptible", 0),
            infected=raw.get("infected", 0),
            recovered=raw.get("recovered", 0),
            dead=raw.get("dead", 0),
            total_population=raw.get("total_population", 3000),
            survival_rate=raw.get("survival_rate", 1.0),
        )

    def get_trajectory(self):
        """Return trajectory from the underlying env for grading."""
        if self.env is None:
            return []
        return self.env.get_trajectory()
