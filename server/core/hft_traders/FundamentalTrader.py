try:
    from hft.server.core.hft_traders.Trader import Trader
except ModuleNotFoundError:
    from server.core.hft_traders.Trader import Trader


class FundamentalTrader(Trader):
    def __init__(
        self, agent_id, fundamental=100.0, threshold=0.05, max_order=200, tick=0.01
    ):
        super().__init__(agent_id, base_size=None, tick=tick)
        self.fundamental = fundamental
        self.threshold = threshold
        self.max_order = max_order

    def act(self, book, t):
        if book.mid is None:
            return []

        deviation = self.fundamental - book.mid

        if abs(deviation) < self.threshold:
            return []

        size = int(min(self.max_order, abs(deviation) * 1000))
        size = max(size, 50)

        if deviation > 0:
            return [{"type": "market", "side": "bid", "size": size}]
        else:
            return [{"type": "market", "side": "ask", "size": size}]

    def on_fill(self, side, price, size):
        pass
