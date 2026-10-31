"""Credential-free, network-blocked test runner for CI and local checks.

Discovery is explicit and safe: every candidate module is parsed (never imported) and excluded, with a visible reason,
if it executes calls at import time (the legacy API smoke scripts do). Surviving modules run with sockets blocked and an
empty UPSTOX_ACCESS_TOKEN, so an accidental network or authenticated call fails loudly instead of reaching the vendor.
"""
import argparse
import ast
import os
import socket
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Calls that reach the network, credentials, the filesystem's real data, or other processes. The socket block in
# block_network() is the backstop for anything this static screen cannot see.
DENIED_TOP_LEVEL_CALLS = {"get", "post", "put", "request", "urlopen", "make_request", "get_expiries", "get_expired_contracts",
                          "get_nifty_candles", "get_expired_candles", "load_dotenv", "getenv", "read_parquet", "read_csv",
                          "read_json", "open", "connect", "run", "system", "Popen", "input", "verified_session"}


def _call_name(node):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "<call>")


def import_time_calls(path):
    """Names of calls executed at import time outside defs/classes and the __main__ guard."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                                            # docstring
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = _call_name(child)
                if name in DENIED_TOP_LEVEL_CALLS:
                    found.append(name)
    return sorted(set(found))


def classify(paths):
    safe, excluded = [], []
    for path in sorted(paths):
        calls = import_time_calls(path)
        (excluded if calls else safe).append((path, calls))
    return safe, excluded


def block_network():
    def refuse(self, *args, **kwargs):
        raise RuntimeError("Network access is disabled in the offline test run")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    os.environ["UPSTOX_ACCESS_TOKEN"] = ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Show discovery decisions and exit")
    parser.add_argument("--pattern", default="test*.py")
    args = parser.parse_args(argv)
    candidates = [p for folder in (ROOT / "tests", ROOT / "src") for p in folder.glob(args.pattern)]
    safe, excluded = classify(candidates)
    for path, calls in excluded:
        print(f"EXCLUDED {path.relative_to(ROOT)}: executes {', '.join(calls)} at import time")
    if args.list:
        for path, _ in safe:
            print(f"INCLUDED {path.relative_to(ROOT)}")
        return 0
    block_network()
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for path, _ in safe:
        suite.addTests(loader.loadTestsFromName(path.stem))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(f"Ran {result.testsRun} tests from {len(safe)} modules; {len(excluded)} excluded as import-time online scripts")
    return 0 if result.wasSuccessful() and result.testsRun > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
