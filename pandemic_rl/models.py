"""
Pydantic Typed Models for the Pandemic RL Environment
=====================================================
Uses OpenEnv's base Action and Observation classes for SDK compliance.
"""

from typing import List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class PandemicAction(Action):
    """Single discrete action: integer 0-6."""
    action: int  # 0=nothing, 1-3=quarantine city, 4-6=vaccinate city


class PandemicObservation(Observation):
    """Full observation from the pandemic environment."""
    # 12 floats: per city [S, I, R, D] normalized /1000
    city_data: List[float] = Field(default_factory=list)
    susceptible: int = 0
    infected: int = 0
    recovered: int = 0
    dead: int = 0
    survival_rate: float = 0.0
    step: int = 0
    done: bool = False
    # Some OpenEnv utilities return reward alongside observation; we carry it
    # so client/server/inference can round-trip without schema errors.
    reward: Optional[float] = None
