"""
OpenEnv Client for Pandemic RL
===============================
Properly implements the OpenEnv EnvClient ABC (WebSocket-based).
Used by inference.py to interact with the dockerized environment.
"""

from typing import Any, Dict

from pydantic import Field
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from pandemic_rl.models import PandemicAction, PandemicObservation


class PandemicState(State):
    """State type for the pandemic environment client."""
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


class PandemicEnvClient(EnvClient[PandemicAction, PandemicObservation, PandemicState]):
    """
    WebSocket client for the Pandemic RL OpenEnv server.

    Usage:
        async with PandemicEnvClient(base_url="http://localhost:8000") as env:
            result = await env.reset(task_name="TaskEasy")
            while not result.done:
                action = PandemicAction(action=1)
                result = await env.step(action)
    """

    def _step_payload(self, action: PandemicAction) -> Dict[str, Any]:
        """Convert PandemicAction to the JSON dict expected by the server."""
        return {"action": action.model_dump()}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PandemicObservation]:
        """Convert server response to StepResult with typed observation."""
        obs_data = payload.get("observation", payload)

        if isinstance(obs_data, dict):
            observation = PandemicObservation(**obs_data)
        else:
            # Fallback for raw list format
            observation = PandemicObservation(
                city_data=obs_data[:12] if isinstance(obs_data, list) else [],
                done=payload.get("done", False),
                reward=payload.get("reward"),
            )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PandemicState:
        """Convert server state response to typed PandemicState."""
        return PandemicState(**payload)
