#!/usr/bin/env python3
"""
call_stack.py

A robust, scalable, and maintainable simulation of a call stack (LIFO) that
mirrors JavaScript’s synchronous execution model. The implementation strictly
adheres to PEP‑8, leverages type hints, and provides comprehensive error
handling. Example usage at the bottom demonstrates end‑to‑end behavior.

Author: ChatGPT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

# --------------------------------------------------------------------------- #
# Logging Configuration                                                       #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Exceptions                                                                  #
# --------------------------------------------------------------------------- #


class StackOverflowError(RuntimeError):
    """Raised when the stack grows beyond its maximum allowed depth."""


class StackUnderflowError(RuntimeError):
    """Raised when attempting to pop from an empty stack."""


# --------------------------------------------------------------------------- #
# Data Structures                                                             #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CallFrame:
    """
    Represents a single frame in the call stack.

    Attributes
    ----------
    function_name : str
        Name of the function being executed.
    args : tuple[Any, ...]
        Positional arguments supplied to the function.
    kwargs : dict[str, Any]
        Keyword arguments supplied to the function.
    """

    function_name: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.function_name}{self.args}{self.kwargs}"


# --------------------------------------------------------------------------- #
# Core CallStack Implementation                                               #
# --------------------------------------------------------------------------- #


class CallStack:
    """
    LIFO call stack with configurable maximum depth, mirroring JavaScript’s
    synchronous execution context.
    """

    __slots__ = ("_frames", "_max_depth")

    def __init__(self, max_depth: int = 1_000) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        self._frames: List[CallFrame] = []
        self._max_depth: int = max_depth

    # --------------------------------------------------------------------- #
    # Stack Operations                                                      #
    # --------------------------------------------------------------------- #

    def push(self, frame: CallFrame) -> None:
        """Push a new call frame onto the stack."""
        if self.size >= self._max_depth:
            raise StackOverflowError("Maximum call stack size exceeded")
        self._frames.append(frame)
        logger.debug("PUSH %s | Depth: %d", frame, self.size)

    def pop(self) -> CallFrame:
        """Pop the top‑most call frame off the stack."""
        if not self._frames:
            raise StackUnderflowError("Cannot pop from an empty stack")
        frame = self._frames.pop()
        logger.debug("POP  %s | Depth: %d", frame, self.size)
        return frame

    def peek(self) -> Optional[CallFrame]:
        """Return the top‑most frame without removing it."""
        return self._frames[-1] if self._frames else None

    # --------------------------------------------------------------------- #
    # Properties                                                            #
    # --------------------------------------------------------------------- #

    @property
    def size(self) -> int:
        """Current depth of the stack."""
        return len(self._frames)

    @property
    def is_empty(self) -> bool:
        """Check whether the stack is empty."""
        return not self._frames

    # --------------------------------------------------------------------- #
    # Utility                                                                #
    # --------------------------------------------------------------------- #

    def clear(self) -> None:
        """Remove all frames from the stack."""
        self._frames.clear()
        logger.debug("CLEAR | Depth: 0")

    def __str__(self) -> str:
        return " -> ".join(str(frame) for frame in reversed(self._frames))


# --------------------------------------------------------------------------- #
# Decorator for Automatic Stack Handling                                      #
# --------------------------------------------------------------------------- #

_GLOBAL_STACK = CallStack(max_depth=10_000)  # Global simulation of JS call stack


def with_call_stack(func):
    """
    Decorator that automatically pushes a CallFrame onto the global stack upon
    function entry and pops it on exit (even under exceptions).
    """

    def wrapper(*args, **kwargs):
        frame = CallFrame(func.__name__, args, kwargs)
        _GLOBAL_STACK.push(frame)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            _GLOBAL_STACK.pop()

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# --------------------------------------------------------------------------- #
# Example: Recursive Factorial Using the Call Stack                           #
# --------------------------------------------------------------------------- #

@with_call_stack
def factorial(n: int) -> int:
    """
    Compute factorial recursively, leveraging the CallStack decorator to show
    how each recursive call is managed.
    """
    if n < 0:
        raise ValueError("n must be non‑negative")
    return 1 if n in (0, 1) else n * factorial(n - 1)


# --------------------------------------------------------------------------- #
# Main (Demonstration)                                                        #
# --------------------------------------------------------------------------- #

def _demo() -> None:
    """Demonstrate call stack behavior and error handling."""
    try:
        number = 5
        logger.info("Calculating factorial(%d)", number)
        result = factorial(number)
        logger.info("Result: %d", result)
        logger.info("Final stack depth: %d", _GLOBAL_STACK.size)
    except (StackOverflowError, StackUnderflowError, ValueError) as exc:
        logger.error("Error: %s", exc)


if __name__ == "__main__":
    _demo()