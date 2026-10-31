"""Futures carry no purchase notional; only realized P&L/fees affect cash."""

from dataclasses import dataclass


@dataclass
class FuturesPosition:
    quantity: int = 0
    average_fill: float = 0.

    def unrealized(self, mark):
        return self.quantity * (mark - self.average_fill)

    def trade(self, change, fill):
        old = self.quantity
        new = old + change
        realized = 0.
        if old == 0 or old * change >= 0:
            self.average_fill = (abs(old) * self.average_fill + abs(change) * fill) / abs(new) if new else 0.
        else:
            closing = min(abs(old), abs(change))
            realized = closing * (1 if old > 0 else -1) * (fill - self.average_fill)
            if new == 0:
                self.average_fill = 0.
            elif old * new < 0:
                self.average_fill = fill
        self.quantity = new
        return realized
