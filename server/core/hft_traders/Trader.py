from typing import Optional


class Trader:
    def __init__(
        self,
        agent_id: int,
        base_size: Optional[int | None] = 100,
        tick: int = 0.01,
        inventory: int = 0,
        cash: float = 0.0,
    ):
        self.agent_id = agent_id
        self.base_size = base_size
        self.tick = tick

        self.active_orders = []
        self.inventory = inventory
        self.cash = cash

    def act(self, book, t):
        pass

    def on_fill(self, side, size, price):
        """Called by exchange when one of our orders is filled."""
        if side == "bid":
            self.inventory += size
            self.cash -= size * price
        else:
            self.inventory -= size
            self.cash += size * price
