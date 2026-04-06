# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Hft Environment.

The hft environment is a simple test environment that echoes back messages.
"""

from enum import Enum
from typing import List, Optional
from typing_extensions import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

tasks = ["basic_execution", "false_signal", "conflicting_signal", "flash_crash"]


class HftAction(Action):
    """Action for the Hft environment.

    Attributes:
        limit_buy_price: Price for limit buy orders
        limit_buy_size: Size for limit buy orders
        limit_ask_price: Price for limit sell orders
        limit_ask_size: Size for limit sell orders
        market_buy_size: Size for market buy orders
        market_ask_size: Size for market sell orders
        cancel_order_id: Order ID for cancel actions
    """

    limit_buy_price: Optional[float] = Field(
        default=None, description="Price for limit buy orders"
    )
    limit_buy_size: Optional[int] = Field(
        default=None, description="Size for limit buy orders"
    )
    limit_ask_price: Optional[float] = Field(
        default=None, description="Price for limit sell orders"
    )
    limit_ask_size: Optional[int] = Field(
        default=None, description="Size for limit sell orders"
    )
    market_buy_size: Optional[int] = Field(
        default=None, description="Size for market buy orders"
    )
    market_ask_size: Optional[int] = Field(
        default=None, description="Size for market sell orders"
    )
    cancel_order_id: Optional[str] = Field(
        default=None, description="Order ID for cancel actions"
    )


class HftObservation(Observation):
    """Observation from the Hft environment.

    Attributes:
        time: Current simulation time
        reward: The reward received from current book order
        done: Flag indicating if the episode has ended
        history: Historical order book states leading up to the current observation
        active_orders: List of active order IDs placed by the agent
    """

    time: float = Field(default=0.0, description="Current time in the simulation")
    reward: float = Field(
        default=0.0, description="The reward received from current book order"
    )
    done: bool = Field(
        default=False, description="Flag indicating if the episode has ended"
    )
    history: list = Field(
        default_factory=list,
        description="Historical order book states leading up to the current observation",
    )
    active_orders: list = Field(
        default_factory=list, description="List of active order IDs placed by the agent"
    )


class HftState(State):
    """Extended state for the Hft environment.

    Attributes:
        task_name: The name of the current task or scenario (BASIC_EXEC, FALSE_SIGNAL, CONFLICTING_SIGNAL, FLASH_CRASH)
        episode_id: Unique identifier for the current episode
        active_orders: List of active order IDs placed by the agent
        step_count: The current step count within the episode
        max_inventory: Maximum inventory limit for the agent
        inventory: Number of shares held
        cash: Amount of money in the account
    """

    task_name: str = Field(
        default="basic_execution",
        description="The name of the current task or scenario",
    )
    episode_id: str = Field(
        default="", description="Unique identifier for the current episode"
    )
    active_orders: list = Field(
        default_factory=list, description="List of active order IDs placed by the agent"
    )
    step_count: int = Field(
        default=0, description="The current step count within the episode"
    )
    max_inventory: int = Field(
        default=1000, description="Maximum inventory limit for the agent"
    )
    inventory: int = Field(default=0, description="Number of shares held")
    cash: float = Field(default=0.0, description="Amount of money in the account")
