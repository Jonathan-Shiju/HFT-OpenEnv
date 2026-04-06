# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hft Environment Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from hft.models import HftAction, HftObservation, HftState
except ModuleNotFoundError:
    from models import HftAction, HftObservation, HftState


class HftEnv(EnvClient[HftAction, HftObservation, HftState]):
    """
    Client for the Hft Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with HftEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.mid)
        ...
        ...     action = HftAction(market_buy_size=10)
        ...     result = client.step(action)
        ...     print(result.observation.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = HftEnv.from_docker_image("hft-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     action = HftAction(limit_ask_price=101.5, limit_ask_size=5)
        ...     result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: HftAction) -> Dict:
        """
        Convert HftAction to JSON payload for step message.

        Args:
            action: HftAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "limit_buy_price": action.limit_buy_price,
            "limit_buy_size": action.limit_buy_size,
            "limit_ask_price": action.limit_ask_price,
            "limit_ask_size": action.limit_ask_size,
            "market_buy_size": action.market_buy_size,
            "market_ask_size": action.market_ask_size,
            "cancel_order_id": action.cancel_order_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[HftObservation]:
        """
        Parse server response into StepResult[HftObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with HftObservation
        """
        obs_data = payload.get("observation", {})
        observation = HftObservation(
            time=obs_data.get("time", 0.0),
            reward=payload.get("reward", 0.0),
            done=obs_data.get("done", False),
            history=obs_data.get("history", []),
            active_orders=obs_data.get("active_orders", []),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> HftState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode and portfolio fields
        """
        return HftState(
            task_name=payload.get("task_name"),
            episode_id=payload.get("episode_id"),
            active_orders=payload.get("active_orders", []),
            step_count=payload.get("step_count", 0),
            max_inventory=payload.get("max_inventory", 0),
            inventory=payload.get("inventory", 0),
            cash=payload.get("cash", 0.0),
        )
