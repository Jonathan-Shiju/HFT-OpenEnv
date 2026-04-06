from collections import defaultdict


class OrderBook:
    def __init__(self, arrival_price=100.0):
        self.bids = defaultdict(list)
        self.asks = defaultdict(list)
        self.orders = {}
        self.arrival_price = arrival_price
        self.next_id = 0
        self.total_traded_volume = 0.0

    def add_order(self, side, price, size, agent_id):
        """
        Add a limit order and immediately match it against the opposite side.
        Returns (order_id, fills) where:
        - order_id is the ID of the resting order (or None if fully filled)
        - fills is a list of fill dicts with keys: price, size, buyer, seller
        """
        order_id = self.next_id
        self.next_id += 1
        order = {
            "id": order_id,
            "side": side,
            "price": price,
            "size": size,
            "agent_id": agent_id,
        }
        self.orders[order_id] = order
        remaining = size
        fills = []

        if side == "bid":
            while (
                remaining > 0 and self.best_ask is not None and price >= self.best_ask
            ):
                current_best_ask = self.best_ask

                if current_best_ask not in self.asks or not self.asks[current_best_ask]:
                    if current_best_ask in self.asks:
                        del self.asks[current_best_ask]
                    continue

                ask_order = self.asks[current_best_ask][0]
                fill = min(remaining, ask_order["size"])

                fills.append(
                    {
                        "price": ask_order["price"],
                        "size": fill,
                        "buyer": agent_id,
                        "seller": ask_order["agent_id"],
                    }
                )

                self.total_traded_volume += fill
                remaining -= fill
                ask_order["size"] -= fill

                if ask_order["size"] == 0:
                    self.cancel_order(ask_order["id"])
                    if not self.asks[current_best_ask]:
                        del self.asks[current_best_ask]

            if remaining > 0:
                order["size"] = remaining
                if price not in self.bids:
                    self.bids[price] = []
                self.bids[price].append(order)
                return order_id, fills
            else:
                if order_id in self.orders:
                    del self.orders[order_id]
                return None, fills

        else:  # side == 'ask'
            while (
                remaining > 0 and self.best_bid is not None and price <= self.best_bid
            ):
                current_best_bid = self.best_bid

                if current_best_bid not in self.bids or not self.bids[current_best_bid]:
                    if current_best_bid in self.bids:
                        del self.bids[current_best_bid]
                    continue

                bid_order = self.bids[current_best_bid][0]
                fill = min(remaining, bid_order["size"])

                fills.append(
                    {
                        "price": bid_order["price"],
                        "size": fill,
                        "buyer": bid_order["agent_id"],
                        "seller": agent_id,
                    }
                )

                self.total_traded_volume += fill
                remaining -= fill
                bid_order["size"] -= fill

                if bid_order["size"] == 0:
                    self.cancel_order(bid_order["id"])
                    if not self.bids[current_best_bid]:
                        del self.bids[current_best_bid]

            if remaining > 0:
                order["size"] = remaining
                if price not in self.asks:
                    self.asks[price] = []
                self.asks[price].append(order)
                return order_id, fills
            else:
                if order_id in self.orders:
                    del self.orders[order_id]
                return None, fills

    def cancel_order(self, order_id):
        if order_id not in self.orders:
            return False
        order = self.orders.pop(order_id)
        level = self.bids if order["side"] == "bid" else self.asks
        level[order["price"]] = [
            o for o in level[order["price"]] if o["id"] != order_id
        ]
        if not level[order["price"]]:
            del level[order["price"]]

    @property
    def best_bid(self):
        return max(self.bids.keys()) if self.bids else None

    @property
    def best_ask(self):
        return min(self.asks.keys()) if self.asks else None

    @property
    def mid(self):
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.arrival_price

    def get_depth(self, levels=5):
        bid_levels = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_levels = sorted(self.asks.keys())[:levels]
        return {
            "bid_prices": bid_levels,
            "bid_sizes": [sum(o["size"] for o in self.bids[p]) for p in bid_levels],
            "ask_prices": ask_levels,
            "ask_sizes": [sum(o["size"] for o in self.asks[p]) for p in ask_levels],
        }
