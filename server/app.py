"""
Pandemic RL — OpenEnv Server
==============================
Uses OpenEnv's create_app() to expose the PandemicEnvironment
as a standard HTTP API with /reset, /step, /state endpoints.
"""

import uvicorn
from openenv.core.env_server.http_server import create_app

from pandemic_rl.models import PandemicAction, PandemicObservation
from pandemic_rl.pandemic_environment import PandemicEnvironment

# Create the OpenEnv-compliant FastAPI app
# create_app(env_factory, action_cls, observation_cls)
app = create_app(
    PandemicEnvironment,
    PandemicAction,
    PandemicObservation,
)


def main():
    """Run the server directly."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()