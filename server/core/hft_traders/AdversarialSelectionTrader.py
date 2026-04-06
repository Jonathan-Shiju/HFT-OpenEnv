import random

try:
    from hft.server.core.hft_traders.Trader import Trader
except ModuleNotFoundError:
    from server.core.hft_traders.Trader import Trader


class AdversarialSelectionTrader(Trader):
    def __init__(
        self,
        agent_id,
        base_size=500,
        n_levels=5,
        spoof_size=20000,
        tick=0.01,
        iceberg_ratio=0.1,
    ):
        super().__init__(agent_id, base_size, tick)
        self.n_levels = n_levels
        self.spoof_size = spoof_size
        self.iceberg_ratio = iceberg_ratio
        self.phase = "predate"
        self.direction = 1
        self.active_orders = []

    def on_fill(self, side, size, price):
        """Update inventory when our 'Iceberg' hook gets hit."""
        super().on_fill(side, size, price)
        self.phase = "flush"

    def _normalize_price(self, price):
        """Fixes the np.float64(99.96000000000001) issue."""
        return round(float(price) / self.tick) * self.tick

    def act(self, book, t):
        actions = []

        if self.active_orders:
            for oid in self.active_orders:
                actions.append({"type": "cancel", "order_id": oid})
            self.active_orders = []

        if not book.bids or not book.asks:
            return actions

        if self.phase == "predate":
            self.direction = 1 if random.random() > 0.5 else -1

            best_bid = self._normalize_price(book.best_bid)
            best_ask = self._normalize_price(book.best_ask)

            for level in range(self.n_levels):
                offset = random.randint(1, 3) * self.tick
                if self.direction == 1:
                    price = self._normalize_price(best_bid - offset)
                    side = "bid"
                else:
                    price = self._normalize_price(best_ask + offset)
                    side = "ask"

                actions.append(
                    {
                        "type": "limit",
                        "side": side,
                        "price": price,
                        "size": self.spoof_size * (level + 1),
                        "tag": "spoof",
                    }
                )

            trap_side = "ask" if self.direction == 1 else "bid"
            trap_price = best_ask if self.direction == 1 else best_bid

            actions.append(
                {
                    "type": "limit",
                    "side": trap_side,
                    "price": self._normalize_price(trap_price),
                    "size": int(self.base_size * self.iceberg_ratio),
                    "tag": "trap",
                }
            )

            self.phase = "wait_for_victim"

        elif self.phase == "wait_for_victim":
            try:
                check_vol = book.bids[0][1] if self.direction == 1 else book.asks[0][1]

                if check_vol > (self.spoof_size * 1.1):
                    self.phase = "flush"
                else:
                    self.phase = "predate"
            except (IndexError, TypeError):
                self.phase = "predate"

        elif self.phase == "flush":
            actions.append(
                {
                    "type": "market",
                    "side": "ask" if self.direction == 1 else "bid",
                    "size": self.base_size * 2,
                }
            )
            self.phase = "predate"

        return actions
