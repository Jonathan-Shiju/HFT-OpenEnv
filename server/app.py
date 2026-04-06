# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Hft Environment.

This module creates an HTTP server that exposes the HftEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e
try:
    from hft.models import HftAction, HftObservation
    from hft.server.hft_environment import HftEnvironment
    import argparse
except ModuleNotFoundError:
    from models import HftAction, HftObservation
    from server.hft_environment import HftEnvironment

import os
from dotenv import load_dotenv


load_dotenv()

import os

print(f"--- DEBUG: MAX_STEPS value is '{os.environ.get('MAX_STEPS')}' ---")
print(
    f"--- DEBUG: Source of variable: {'System' if 'MAX_STEPS' in os.environ else 'Default'} ---"
)

task = os.getenv("TASK", "flash_crash")
max_steps = int(os.getenv("MAX_STEPS", 30))
tick_size = float(os.getenv("TICK_SIZE", 0.01))
inventory = int(os.getenv("INVENTORY", 0))
cash = float(os.getenv("CASH", 0.0))
arrival_price = float(os.getenv("ARRIVAL_PRICE", 100.0))
target_shares = int(os.getenv("TARGET_SHARES", 1000))


def create_HftEnvironment() -> HftEnvironment:
    return HftEnvironment(
        task_name=task,
        max_steps=max_steps,
        tick_size=tick_size,
        inventory=inventory,
        cash=cash,
        arrival_price=arrival_price,
        target_shares=target_shares,
    )


# Create the app with web interface and README integration
app = create_app(
    create_HftEnvironment,
    HftAction,
    HftObservation,
    env_name="hft",
    max_concurrent_envs=5,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m hft.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn hft.server.app:app --workers 4
    """
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if port == 8000 and args.port != 8000:
        port = args.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
