import numpy as np

try:
    from hft.server.core.hft_traders.Trader import Trader
except ModuleNotFoundError:
    from server.core.hft_traders.Trader import Trader


class AvellanedaStoikovMM(Trader):
    def __init__(
        self,
        agent_id,
        gamma=0.1,
        sigma=0.02,
        k=1.5,
        T=1.0,
        n_levels=5,
        base_size=100,
        max_inventory=500,
        tick=0.01,
    ):
        super().__init__(agent_id, base_size, tick)
        self.n_levels = n_levels
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.T = T
        self.max_inventory = max_inventory
        self.t = 0

    def _compute_quotes(self, mid, t):
        time_remaining = max(self.T - t, 0.001)

        inventory_effect = np.clip(
            self.inventory * self.gamma * self.sigma**2 * time_remaining, -0.01, 0.01
        )

        spread = self.gamma * self.sigma**2 * time_remaining + (
            2 / self.gamma
        ) * np.log(1 + self.gamma / self.k)

        reservation = round(mid / self.tick) * self.tick - (inventory_effect)

        raw_bid = reservation - spread / 2
        raw_ask = reservation + spread / 2

        bid_price = np.floor(raw_bid / self.tick) * self.tick
        ask_price = np.ceil(raw_ask / self.tick) * self.tick

        if ask_price <= bid_price:
            ask_price = bid_price + self.tick

        return bid_price, ask_price

    def act(self, book, t):
        """Called each simulation step. Returns list of actions."""
        self.t = t
        actions = []
        mid = book.mid

        for order_id in self.active_orders:
            actions.append({"type": "cancel", "order_id": order_id})
        self.active_orders = []

        best_bid, best_ask = self._compute_quotes(mid, t)

        for level in range(1, self.n_levels + 1):
            size = self.base_size * level

            if self.inventory > self.max_inventory * 0.5:
                bid_size = max(size // 2, 10)
                ask_size = size
            elif self.inventory < -self.max_inventory * 0.5:
                bid_size = size
                ask_size = max(size // 2, 10)
            else:
                bid_size = ask_size = size

            bid_price = np.floor(best_bid / self.tick) * self.tick - (
                (level - 1) * self.tick
            )
            ask_price = np.ceil(best_ask / self.tick) * self.tick + (
                (level - 1) * self.tick
            )

            if ask_price <= bid_price:
                ask_price = bid_price + self.tick

            actions.append(
                {"type": "limit", "side": "bid", "price": bid_price, "size": bid_size}
            )
            actions.append(
                {"type": "limit", "side": "ask", "price": ask_price, "size": ask_size}
            )

        return actions
