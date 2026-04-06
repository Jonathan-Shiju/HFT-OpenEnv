import numpy as np

try:
    from hft.server.core.hft_traders.Trader import Trader
except ModuleNotFoundError:
    from server.core.hft_traders.Trader import Trader


class NoiseTrader(Trader):
    def __init__(
        self,
        agent_id,
        arrival_rate=0.4,
        base_size=30,
        sigma=0.02,
        tick=0.01,
        market_order_prob=0.2,
    ):
        super().__init__(agent_id, base_size=base_size, tick=tick)
        self.arrival_rate = arrival_rate
        self.sigma = sigma
        self.market_order_prob = market_order_prob

    def act(self, book, t):
        actions = []

        for order_id in self.active_orders:
            actions.append({"type": "cancel", "order_id": order_id})
        self.active_orders = []

        if np.random.rand() > self.arrival_rate:
            return actions

        mid = book.mid
        if mid is None:
            return actions

        depth = book.get_depth(levels=5)
        bid_vol = sum(depth["bid_sizes"])
        ask_vol = sum(depth["ask_sizes"])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
        prob_buy = np.clip(0.5 - 0.2 * imbalance, 0.2, 0.8)
        side = "bid" if np.random.rand() < prob_buy else "ask"

        if np.random.rand() < self.market_order_prob:
            size = max(int(np.random.exponential(self.base_size)), 1)
            actions.append({"type": "market", "side": side, "size": size})
            return actions

        size = max(int(self.base_size * np.random.uniform(0.5, 1.5)), 1)
        noise = abs(np.random.normal(0, self.sigma * 5))

        if side == "bid":
            max_bid = (book.best_ask - self.tick) if book.best_ask else mid
            price = round(min(mid - noise, max_bid) / self.tick) * self.tick
        else:
            min_ask = (book.best_bid + self.tick) if book.best_bid else mid
            price = round(max(mid + noise, min_ask) / self.tick) * self.tick

        actions.append({"type": "limit", "side": side, "price": price, "size": size})
        return actions
