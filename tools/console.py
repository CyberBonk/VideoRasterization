"""Tiny colored console helpers for pipeline status output."""
from __future__ import annotations

import sys


GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def color_text(message: str, color: str) -> str:
    return f"{color}{message}{RESET}"


def info(message: str) -> None:
    print(message)


def ok(message: str) -> None:
    print(color_text(message, GREEN))


def warn(message: str) -> None:
    print(color_text(message, YELLOW))


def error(message: str) -> None:
    print(color_text(message, RED), file=sys.stderr)


def status(message: str) -> None:
    if message.startswith(("[ok]", "[done]", "[mode]", "[start]", "[progress]")):
        ok(message)
    elif message.startswith(("[error]", "[fail]")):
        error(message)
    elif message.startswith("[warn]"):
        warn(message)
    else:
        info(message)
