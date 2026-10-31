"""Read-only market data access for the daily runner: live Upstox, and a clearly synthetic replay of stored candles.

Nothing here can place an order. Live quotes and candles use the project's verified HTTPS client. The replay market builds
quotes from stored candle opens with a one-tick spread and unlimited depth, so replay sessions are for software verification
only and are registered in a mode that can never count as forward evidence.
"""
import urllib.parse

import pandas as pd

IST = "Asia/Kolkata"
INDEX_KEY = "NSE_INDEX|Nifty 50"
BASE = "https://api.upstox.com/v3/historical-candle"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]


def _frame(candles):
    frame = pd.DataFrame(candles, columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame.timestamp).dt.tz_convert(IST)
    return frame.sort_values("timestamp").reset_index(drop=True)


class UpstoxMarket:
    live = True

    def __init__(self, request=None):
        if request is None:
            from upstox_api import make_request
            request = make_request
        self.request = request

    def candles(self, key, day, today):
        encoded = urllib.parse.quote(key, safe="")
        url = f"{BASE}/intraday/{encoded}/minutes/5" if pd.Timestamp(day).date() == pd.Timestamp(today).date() \
            else f"{BASE}/{encoded}/minutes/5/{day}/{day}"
        return _frame(self.request(url).get("data", {}).get("candles", []))

    def quotes(self, keys, at=None):
        query = urllib.parse.quote(",".join(keys), safe=",")
        payload = self.request(f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={query}")
        return pd.Timestamp.now(tz="UTC"), payload


class ReplayMarket:
    """Stored candles for one past session; quotes are synthetic (next-bar open +/- one tick, unlimited depth)."""
    live = False

    def __init__(self, frames, tick=.05, size=10_000_000):
        self.frames, self.tick, self.size = frames, tick, size

    def candles(self, key, day, today):
        return self.frames[key]

    def quotes(self, keys, at):
        at = pd.Timestamp(at)
        received = (at + pd.Timedelta(seconds=1)).tz_convert("UTC")
        data = {}
        for key in keys:
            frame = self.frames[key]
            row = frame[frame.timestamp == at]               # the bar that starts when the decision bar completes
            if row.empty:
                continue
            mid = float(row.open.iloc[0])
            data[key] = {"instrument_token": key, "timestamp": (at + pd.Timedelta(seconds=.5)).isoformat(), "last_price": mid,
                         "depth": {"buy": [{"quantity": self.size, "price": round(mid - self.tick, 2), "orders": 1}],
                                   "sell": [{"quantity": self.size, "price": round(mid + self.tick, 2), "orders": 1}]}}
        return received, {"status": "success", "data": data}


def fetch_replay_frames(keys, day, market=None):
    """Download stored candles for a past day (read-only) and return them for ReplayMarket."""
    market = market or UpstoxMarket()
    return {key: market.candles(key, day, today=pd.Timestamp.now(tz=IST).date() + pd.Timedelta(days=3)) for key in keys}
