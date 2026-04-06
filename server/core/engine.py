from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import yaml
import random

try:
    from hft.server.core.OrderBook import OrderBook
    from hft.server.core.hft_traders.AdversarialSelectionTrader import (
        AdversarialSelectionTrader,
    )
    from hft.server.core.hft_traders.AvellanedaStoikovModel import AvellanedaStoikovMM
    from hft.server.core.hft_traders.FundamentalTrader import FundamentalTrader
    from hft.server.core.hft_traders.NoiseTrader import NoiseTrader
    from hft.server.core.hft_traders.Trader import Trader

    from hft.models import HftState, HftAction
except ModuleNotFoundError:
    from server.core.OrderBook import OrderBook
    from server.core.hft_traders.AdversarialSelectionTrader import (
        AdversarialSelectionTrader,
    )
    from server.core.hft_traders.AvellanedaStoikovModel import AvellanedaStoikovMM
    from server.core.hft_traders.FundamentalTrader import FundamentalTrader
    from server.core.hft_traders.NoiseTrader import NoiseTrader
    from server.core.hft_traders.Trader import Trader

    from models import HftState, HftAction


TARGET_PARTICIPATION = 0.10
BANKRUPTCY_THRESHOLD = -5000.0


from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyConfig:
    w_pnl: float
    w_passive: float
    w_urgency: float
    w_terminal: float
    max_raw: float
    step_penalty: float
    w_adverse: float
    edge_threshold_bps: float
    target_participation: float


DIFFICULTY_CONFIGS: Dict[int, DifficultyConfig] = {
    1: DifficultyConfig(1.0, 0.30, 0.010, 1.0, 0.030, 0.000, 0.00, 0.0, 0.05),
    2: DifficultyConfig(1.0, 0.30, 0.025, 1.33, 0.187, 0.0083, 0.166, 5.0, 0.066),
    3: DifficultyConfig(1.0, 0.30, 0.040, 1.66, 0.344, 0.0166, 0.333, 10.0, 0.083),
    4: DifficultyConfig(1.0, 0.30, 0.055, 2.0, 0.500, 0.025, 0.50, 15.0, 0.10),
}


class LOBReward:
    def __init__(
        self,
        target_shares: int,
        arrival_price: float,
        T: int,
        side: str = "bid",
        difficulty: int = 1,
        w_pnl: Optional[float] = None,
        w_passive: Optional[float] = None,
        w_urgency: Optional[float] = None,
        w_terminal: Optional[float] = None,
        max_raw: Optional[float] = None,
    ):
        if difficulty not in DIFFICULTY_CONFIGS:
            raise ValueError(f"difficulty must be 1-4, got {difficulty}")

        self.target = target_shares
        self.arrival_price = arrival_price
        self.T = T
        self.side = side
        self.cfg = DIFFICULTY_CONFIGS[difficulty]

        self.w_pnl = w_pnl if w_pnl is not None else self.cfg.w_pnl
        self.w_passive = w_passive if w_passive is not None else self.cfg.w_passive
        self.w_urgency = w_urgency if w_urgency is not None else self.cfg.w_urgency
        self.w_terminal = w_terminal if w_terminal is not None else self.cfg.w_terminal
        self.max_raw = max_raw if max_raw is not None else self.cfg.max_raw

    def _normalize(self, raw: float) -> float:
        clipped = float(np.clip(raw, -self.max_raw, self.max_raw))
        linear_norm = (clipped + self.max_raw) / (2.0 * self.max_raw)
        k = 15.0
        scaled_reward = 1 / (1 + np.exp(-k * (linear_norm - 0.5)))
        return float(scaled_reward)

    def _execution_edge(self, fill_price: float, current_mid: float) -> float:
        if current_mid <= 0:
            return 0.0
        scale = 10_000.0
        if self.side == "bid":
            return ((current_mid - fill_price) / current_mid) * scale
        return ((fill_price - current_mid) / current_mid) * scale

    def _pnl_component(
        self, fill_price: float, fill_size: float, current_mid: float
    ) -> float:
        if fill_size <= 0 or current_mid <= 0:
            return 0.0
        edge = self._execution_edge(fill_price, current_mid)
        adjusted_edge = edge - self.cfg.edge_threshold_bps
        return self.w_pnl * adjusted_edge * (fill_size / self.target) * 0.0005

    def _passive_component(self, fill_type: str) -> float:
        if fill_type == "passive":
            return self.w_passive * 0.02
        if fill_type == "active":
            return -self.w_passive * 0.01
        return 0.0

    def _progress_component(self, remaining: float, t: float) -> float:
        t_norm = float(np.clip(t, 0.0, 1.0))
        expected_remaining = self.target * (1.0 - t_norm)
        pace_delta = float(
            np.clip((expected_remaining - remaining) / self.target, -1.0, 1.0)
        )
        return self.w_urgency * pace_delta

    def _participation_component(self, participation: float) -> float:
        tp = self.cfg.target_participation
        delta = participation - tp
        if delta < 0:
            return self.w_urgency * delta * 10.0
        return -self.w_urgency * (delta**2)

    def _adverse_component(self, adverse_score: float) -> float:
        if self.cfg.w_adverse <= 0 or adverse_score <= 0.0:
            return 0.0
        return -self.cfg.w_adverse * float(np.clip(adverse_score, 0.0, 1.0)) * 0.10

    def _terminal_component(
        self, remaining: float, participation: float, bankrupt: bool
    ) -> float:
        if bankrupt:
            return -self.max_raw
        raw = 0.0
        if remaining > 0:
            raw -= self.w_terminal * ((remaining / self.target) ** 2)
        if participation < self.cfg.target_participation:
            raw -= self.w_terminal * (self.cfg.target_participation - participation)
        return raw

    def step(
        self,
        fill_price: float,
        fill_size: float,
        fill_type: str,
        remaining: float,
        t: float,
        current_mid: float,
        inventory: float = 0.0,
        participation: float = 0.0,
        adverse_score: float = 0.0,
        terminal: bool = False,
        bankrupt: bool = False,
    ) -> float:
        raw: float = 0.0
        raw += self._pnl_component(fill_price, fill_size, current_mid)
        raw += self._passive_component(fill_type)
        raw += self._progress_component(remaining, t)
        raw += self._participation_component(participation)
        raw += self._adverse_component(adverse_score)
        raw -= self.cfg.step_penalty

        if terminal:
            raw += self._terminal_component(remaining, participation, bankrupt)

        return self._normalize(raw)


class MarketSimulation:
    """Simulate the agent, background traders, order book, and reward flow."""

    def __init__(
        self,
        task_name: str,
        tick_size: float,
        target_shares: int = 1000,
        max_steps: int = 390,
        state: HftState = None,
        arrival_price: float = 100.0,
    ):
        self._task_name = task_name
        self._tick_size = tick_size
        self._target_shares = target_shares
        self._state = state
        self._agent_id = -1
        self._max_steps = max_steps
        self._arrival_price = arrival_price

        self._cumulative_agent_volume = 0.0
        self._cumulative_market_volume = 0.0
        self._agents = []
        self.history = []

        self._spawn_agents()

        self._book = OrderBook(arrival_price=arrival_price)
        self._remaining = float(target_shares)

        self._reward_fn = LOBReward(
            target_shares=target_shares,
            arrival_price=arrival_price,
            T=max_steps,
            difficulty=self._level,
        )

        self._step_fill_price = 0.0
        self._step_fill_size = 0.0
        self._step_fill_type = "none"
        self._step_market_volume = 0.0
        self._step_adversarial_volume = 0.0

        self._current_reward = 0.0
        self._current_participation = 0.0
        self._done = False
        self._bankrupt = False
        self._dt = 1 / self._max_steps
        self._t = 0.0

        self._seed_book()

    def _seed_book(self) -> None:
        """Warm up the book with background trading before the episode starts."""
        seed_steps = int(self._max_steps * 0.2)
        for _ in range(seed_steps):
            self._step_background_only()
        self._record_history()
        self._remaining = float(self._target_shares)

    def _step_background_only(self) -> None:
        """Advance the market using only background agents."""
        for agent in self._agents:
            if agent.agent_id == self._agent_id:
                continue
            actions = agent.act(self._book, self._t)
            for action in actions:
                self._process_action(agent, action, action_type="passive")
        self._t += self._dt

    def _spawn_agents(self) -> None:
        """Instantiate the configured background traders and the learning agent."""
        classes = {
            "AdversarialSelectionTrader": AdversarialSelectionTrader,
            "AvellanedaStoikovModel": AvellanedaStoikovMM,
            "FundamentalTrader": FundamentalTrader,
            "NoiseTrader": NoiseTrader,
        }
        task = self._task_name
        with open(f"server/config/{task}.yaml", "r") as f:
            config = yaml.safe_load(f)

        self._level = config["description"]["level"]
        count = 0
        for agent_cfg in config["agents"]:
            agent_type = agent_cfg["type"]
            agent_count = agent_cfg["count"]
            params = agent_cfg["parameters"]
            for i in range(agent_count):
                agent_params = params[i] if isinstance(params, list) else params
                trader = classes[agent_type](
                    agent_id=f"agent_{count + i}",
                    tick=self._tick_size,
                    **agent_params,
                )
                self._agents.append(trader)
            count += agent_count

        self._agents.append(
            Trader(
                agent_id=self._agent_id,
                tick=self._tick_size,
                inventory=self._state.inventory,
                cash=self._state.cash,
            )
        )

    def _handle_fills(self, fills: list, action_type: str = "passive") -> None:
        """Apply fills to the RL agent and notify counterparties."""
        for fill in fills:
            fill_size = fill["size"]
            fill_price = fill["price"]
            self._step_market_volume += fill_size

            if fill["buyer"] == self._agent_id and fill["seller"] == self._agent_id:
                continue

            if fill["buyer"] == self._agent_id:
                self._state.inventory += fill_size
                self._state.cash -= fill_size * fill_price
                self._remaining -= fill_size
                self._step_fill_price = fill_price
                self._step_fill_size += fill_size
                self._step_fill_type = action_type

                seller = next(
                    (a for a in self._agents if a.agent_id == fill["seller"]), None
                )
                if isinstance(seller, AdversarialSelectionTrader):
                    self._step_adversarial_volume += fill_size
                if seller and seller.agent_id != self._agent_id:
                    seller.on_fill("ask", fill_size, fill_price)

            elif fill["seller"] == self._agent_id:
                self._state.inventory -= fill_size
                self._state.cash += fill_size * fill_price
                self._remaining -= fill_size
                self._step_fill_price = fill_price
                self._step_fill_size += fill_size
                self._step_fill_type = action_type

                buyer = next(
                    (a for a in self._agents if a.agent_id == fill["buyer"]), None
                )
                if isinstance(buyer, AdversarialSelectionTrader):
                    self._step_adversarial_volume += fill_size
                if buyer and buyer.agent_id != self._agent_id:
                    buyer.on_fill("bid", fill_size, fill_price)

            else:
                buyer = next(
                    (a for a in self._agents if a.agent_id == fill["buyer"]), None
                )
                if buyer:
                    buyer.on_fill("bid", fill_size, fill_price)
                seller = next(
                    (a for a in self._agents if a.agent_id == fill["seller"]), None
                )
                if seller:
                    seller.on_fill("ask", fill_size, fill_price)

    def _process_action(self, agent, action: dict, action_type: str) -> None:
        """Route a single action through the book and update agent state."""
        if action["type"] == "cancel":
            self._book.cancel_order(action["order_id"])
            if agent.agent_id == self._agent_id:
                self._state.active_orders = [
                    order
                    for order in self._state.active_orders
                    if str(order["id"]) != action["order_id"]
                ]

        elif action["type"] == "limit":
            order_id, fills = self._book.add_order(
                action["side"],
                action["price"],
                action["size"],
                agent.agent_id,
            )
            f_type = action_type if agent.agent_id == self._agent_id else "passive"
            if agent.agent_id == self._agent_id and order_id is not None:
                self._state.active_orders.append(
                    {
                        "id": order_id,
                        "side": action["side"],
                        "price": action["price"],
                        "size": action["size"],
                    }
                )
            self._handle_fills(fills, action_type=f_type)
            if order_id is not None and hasattr(agent, "active_orders"):
                agent.active_orders.append(order_id)

        elif action["type"] == "market":
            extreme = 1e9 if action["side"] == "bid" else 0.0
            order_id, fills = self._book.add_order(
                action["side"],
                extreme,
                action["size"],
                agent.agent_id,
            )
            f_type = "active" if agent.agent_id == self._agent_id else "passive"
            self._handle_fills(fills, action_type=f_type)
            if order_id is not None:
                self._book.cancel_order(order_id)

    def _record_history(self) -> None:
        """Snapshot the current book state and episode metrics."""
        depth = self._book.get_depth(levels=5)
        clean_bids = [p.item() for p in np.round(depth["bid_prices"], 2)]
        clean_asks = [p.item() for p in np.round(depth["ask_prices"], 2)]
        bids = list(zip(clean_bids, depth["bid_sizes"]))
        asks = list(zip(clean_asks, depth["ask_sizes"]))

        spread = (
            float(round(self._book.best_ask - self._book.best_bid, 4))
            if self._book.best_ask and self._book.best_bid
            else None
        )

        self.history.append(
            {
                "t": round(self._t, 4),
                "mid": float(round(self._book.mid, 4)) if self._book.mid else None,
                "spread": spread,
                "bids": bids,
                "asks": asks,
                "fills": (
                    {
                        "p": float(self._step_fill_price),
                        "s": float(self._step_fill_size),
                        "type": self._step_fill_type,
                    }
                    if self._step_fill_size > 0
                    else None
                ),
                "reward": round(self._current_reward, 3),
                "participation": round(self._current_participation, 4),
                "inv": self._state.inventory,
                "cash": round(self._state.cash, 4),
                "remaining": self._remaining,
                "bankrupt": bool(self._bankrupt),
            }
        )

    def step(self, agent_actions: Optional[HftAction]) -> None:
        """Advance the full simulation by one step."""
        self._step_fill_price = 0.0
        self._step_fill_size = 0.0
        self._step_fill_type = "none"
        self._step_market_volume = 0.0
        self._step_adversarial_volume = 0.0

        active_agents = list(self._agents)
        random.shuffle(active_agents)

        for agent in active_agents:
            if agent.agent_id != self._agent_id:
                actions = agent.act(self._book, self._t)
                action_type = "passive"
            else:
                if agent_actions is None:
                    continue
                actions = agent_actions
                action_type = self._infer_fill_type(agent_actions)
            for action in actions:
                self._process_action(agent, action, action_type)

        self._cumulative_market_volume += self._step_market_volume
        self._cumulative_agent_volume += self._step_fill_size

        self._current_participation = (
            self._cumulative_agent_volume / self._cumulative_market_volume
            if self._cumulative_market_volume > 0
            else 0.0
        )

        if agent_actions is not None:
            current_mid = self._book.mid or self._arrival_price
            unrealized = self._state.inventory * (current_mid - self._arrival_price)
            self._bankrupt = bool(
                (self._state.cash + unrealized) < BANKRUPTCY_THRESHOLD
            )
            is_terminal = (self._t >= (1.0 - self._dt)) or self._bankrupt

            if is_terminal and self._state.inventory != 0:
                sprd = (
                    (self._book.best_ask - self._book.best_bid)
                    if (self._book.best_ask and self._book.best_bid)
                    else 0.1
                )
                self._state.cash -= abs(self._state.inventory) * sprd * 5

            adverse_score = (
                self._step_adversarial_volume / self._step_fill_size
                if self._step_fill_size > 0
                else 0.0
            )

            self._current_reward = self._reward_fn.step(
                fill_price=self._step_fill_price,
                fill_size=self._step_fill_size,
                fill_type=self._step_fill_type,
                remaining=max(self._remaining, 0.0),
                t=self._t if self._t <= 1 else 1,
                current_mid=current_mid,
                participation=self._current_participation,
                adverse_score=adverse_score,
                terminal=is_terminal,
                bankrupt=self._bankrupt,
                inventory=self._state.inventory,
            )

            if is_terminal:
                self._done = True

        self._t += self._dt
        self._record_history()

    def _infer_fill_type(self, actions: List[HftAction]) -> str:
        """Classify the agent's action batch as passive or active."""
        for action in actions:
            if action["type"] == "market":
                return "active"
        return "passive"

    @property
    def reward(self) -> float:
        """Return the most recently computed reward."""
        return self._current_reward

    @property
    def done(self) -> bool:
        """Return whether the episode has terminated."""
        return self._done

    @property
    def bankrupt(self) -> bool:
        """Return whether the agent crossed the bankruptcy threshold."""
        return self._bankrupt

    @property
    def remaining(self) -> float:
        """Return the remaining target quantity to execute."""
        return self._remaining

    @property
    def participation(self) -> float:
        """Return the agent's participation rate in market volume."""
        return self._current_participation

    @property
    def dt(self) -> float:
        """Return the simulation time step."""
        return self._dt

    @property
    def t(self) -> float:
        """Return the current simulation time."""
        return self._t

    @property
    def level(self) -> int:
        """Return the configured difficulty level."""
        return self._level

    @property
    def book(self) -> OrderBook:
        """Return the active order book instance."""
        return self._book
