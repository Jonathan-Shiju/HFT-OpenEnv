# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hft Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""


from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from hft.models import HftAction, HftObservation, HftState
    from hft.server.core.engine import MarketSimulation
except ModuleNotFoundError:
    from models import HftAction, HftObservation, HftState
    from server.core.engine import MarketSimulation


class HftEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = HftEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Hft environment ready!"
        >>>
        >>> obs = env.step(HftAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        task_name: Optional[str] = "flash_crash",
        max_steps: Optional[int] = 390,
        tick_size: Optional[float] = 0.01,
        inventory: Optional[int] = 0,
        cash: Optional[float] = 0.0,
        arrival_price: Optional[float] = 100.0,
        target_shares: Optional[int] = 1000,
    ):
        self._task_name = task_name
        self._max_steps = max_steps
        self._tick_size = tick_size
        self._start_inventory = inventory
        self._start_cash = cash
        self._max_inventory = target_shares * 3
        self._arrival_price = arrival_price
        self._target_shares = target_shares

        self._reset_count = 0
        self.reset()

    def reset(
        self,
        task_name: Optional[str] = None,
        max_steps: Optional[int] = None,
        tick_size: Optional[float] = None,
        inventory: Optional[int] = None,
        cash: Optional[float] = None,
        target_shares: Optional[int] = None,
        arrival_price: Optional[float] = None,
    ) -> HftObservation:
        self._task_name = task_name or self._task_name
        self._max_steps = max_steps or self._max_steps
        self._tick_size = tick_size or self._tick_size
        self._start_inventory = (
            inventory if inventory is not None else self._start_inventory
        )
        self._start_cash = cash if cash is not None else self._start_cash
        self._target_shares = (
            target_shares if target_shares is not None else self._target_shares
        )
        arrival_price = (
            arrival_price if arrival_price is not None else self._arrival_price
        )

        self._state = HftState(
            task_name=self._task_name,
            episode_id=str(uuid4()),
            step_count=0,
            max_inventory=self._max_inventory,
            inventory=self._start_inventory,
            cash=self._start_cash,
            active_orders=[],
        )

        self.sim = MarketSimulation(
            task_name=self._task_name,
            max_steps=self._max_steps,
            tick_size=self._tick_size,
            arrival_price=self._arrival_price,
            state=self._state,
            target_shares=self._target_shares,
        )

        self._reset_count += 1

        history = self.sim.history

        return HftObservation(
            time=round(history[-1]["t"], 4),
            reward=self.sim.reward,
            done=self.sim.done,
            history=history,
            active_orders=[],
        )

    def step(self, actions: HftAction) -> HftObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: HftAction containing the message to echo

        Returns:
            HftObservation with the echoed message and its length
        """
        self._state.step_count += 1

        if actions is not None:
            limit_buy_action = (
                {
                    "type": "limit",
                    "side": "buy",
                    "price": actions.limit_buy_price,
                    "size": actions.limit_buy_size,
                }
                if actions.limit_buy_price and actions.limit_buy_size
                else None
            )

            limit_ask_action = (
                {
                    "type": "limit",
                    "side": "ask",
                    "price": actions.limit_ask_price,
                    "size": actions.limit_ask_size,
                }
                if actions.limit_ask_price and actions.limit_ask_size
                else None
            )

            market_buy_action = (
                {
                    "type": "market",
                    "side": "buy",
                    "size": actions.market_buy_size,
                }
                if actions.market_buy_size
                else None
            )

            market_ask_action = (
                {
                    "type": "market",
                    "side": "ask",
                    "size": actions.market_ask_size,
                }
                if actions.market_ask_size
                else None
            )

            cancel_action = (
                {
                    "type": "cancel",
                    "order_id": actions.cancel_order_id,
                }
                if actions.cancel_order_id
                else None
            )

            actions = [
                action
                for action in [
                    limit_buy_action,
                    limit_ask_action,
                    market_buy_action,
                    market_ask_action,
                    cancel_action,
                ]
                if action is not None
            ]

        # print(actions)
        self.sim.step(actions)

        lookback_window = {
            "1": 3,
            "2": 10,
            "3": 10,
            "4": 10,
        }

        history = self.sim.history
        lookback_history = history[-lookback_window[f"{self.sim.level}"] :]

        return HftObservation(
            time=round(history[-1]["t"], 4),
            reward=self.sim.reward,
            done=self.sim.done,
            history=lookback_history,
            active_orders=self._state.active_orders,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
