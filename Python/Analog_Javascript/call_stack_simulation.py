"""
Module: call_stack_simulation.py

Description:
    This module provides a detailed, robust, and scalable simulation of a Call Stack,
    adhering strictly to PEP-8 standards. The CallStack class models the LIFO (Last-In, First-Out)
    behavior of function invocation and return, similar to JavaScript's call stack.
    It supports synchronous function call simulation, stack overflow/underflow handling,
    and is optimized for large-scale usage.

Author: World #1 Coder (as per task context)
"""

from typing import Any, Callable, List, Optional, Dict


class CallStackOverflowError(Exception):
    """Exception raised when the call stack exceeds its maximum allowed size."""
    pass


class CallStackUnderflowError(Exception):
    """Exception raised when attempting to pop from an empty call stack."""
    pass


class CallFrame:
    """
    Represents a single frame in the call stack.

    Attributes:
        function_name (str): Name of the function.
        args (tuple): Arguments passed to the function.
        kwargs (dict): Keyword arguments passed to the function.
    """

    def __init__(self, function_name: str, args: tuple, kwargs: dict) -> None:
        self.function_name: str = function_name
        self.args: tuple = args
        self.kwargs: dict = kwargs

    def __repr__(self) -> str:
        return (
            f"CallFrame(function_name={self.function_name!r}, "
            f"args={self.args!r}, kwargs={self.kwargs!r})"
        )


class CallStack:
    """
    Simulates a Call Stack (LIFO) for synchronous function execution.

    Attributes:
        max_size (int): Maximum allowed stack size to prevent overflow.
    """

    def __init__(self, max_size: int = 1024) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        self._stack: List[CallFrame] = []
        self.max_size: int = max_size

    def push(self, frame: CallFrame) -> None:
        """
        Pushes a new call frame onto the stack.

        Args:
            frame (CallFrame): The call frame to push.

        Raises:
            CallStackOverflowError: If the stack exceeds max_size.
        """
        if len(self._stack) >= self.max_size:
            raise CallStackOverflowError(
                f"Call stack overflow: maximum size {self.max_size} reached."
            )
        self._stack.append(frame)

    def pop(self) -> CallFrame:
        """
        Pops the top call frame from the stack.

        Returns:
            CallFrame: The popped call frame.

        Raises:
            CallStackUnderflowError: If the stack is empty.
        """
        if not self._stack:
            raise CallStackUnderflowError("Call stack underflow: stack is empty.")
        return self._stack.pop()

    def peek(self) -> Optional[CallFrame]:
        """
        Returns the top call frame without removing it.

        Returns:
            Optional[CallFrame]: The top call frame or None if stack is empty.
        """
        if not self._stack:
            return None
        return self._stack[-1]

    def is_empty(self) -> bool:
        """
        Checks if the call stack is empty.

        Returns:
            bool: True if empty, False otherwise.
        """
        return not self._stack

    def size(self) -> int:
        """
        Returns the current size of the call stack.

        Returns:
            int: Number of frames in the stack.
        """
        return len(self._stack)

    def clear(self) -> None:
        """
        Clears the call stack.
        """
        self._stack.clear()

    def __repr__(self) -> str:
        return f"CallStack(size={self.size()}, max_size={self.max_size})"


class CallStackSimulator:
    """
    Simulates synchronous function calls using a CallStack.

    Methods:
        register_function: Registers a function for simulation.
        call: Simulates calling a registered function.
    """

    def __init__(self, max_stack_size: int = 1024) -> None:
        self.call_stack: CallStack = CallStack(max_stack_size)
        self._functions: Dict[str, Callable[..., Any]] = {}

    def register_function(self, func: Callable[..., Any]) -> None:
        """
        Registers a function for simulation.

        Args:
            func (Callable): The function to register.

        Raises:
            ValueError: If function name is already registered.
        """
        func_name = func.__name__
        if func_name in self._functions:
            raise ValueError(f"Function '{func_name}' is already registered.")
        self._functions[func_name] = func

    def call(self, function_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Simulates calling a registered function.

        Args:
            function_name (str): Name of the function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: The return value of the function.

        Raises:
            KeyError: If function is not registered.
            CallStackOverflowError: If stack overflows.
            Exception: Propagates any exception from the function.
        """
        if function_name not in self._functions:
            raise KeyError(f"Function '{function_name}' is not registered.")

        frame = CallFrame(function_name, args, kwargs)
        self.call_stack.push(frame)
        try:
            result = self._functions[function_name](
                *args, **kwargs
            )
        except Exception as exc:
            # Pop the frame before propagating the exception
            self.call_stack.pop()
            raise exc
        self.call_stack.pop()
        return result

    def stack_trace(self) -> List[CallFrame]:
        """
        Returns the current stack trace.

        Returns:
            List[CallFrame]: List of call frames from bottom to top.
        """
        return list(self.call_stack._stack)

    def clear_stack(self) -> None:
        """
        Clears the call stack.
        """
        self.call_stack.clear()


# Example usage and demonstration
if __name__ == "__main__":
    from typing import NoReturn

    simulator = CallStackSimulator(max_stack_size=100)

    def factorial(n: int) -> int:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0 or n == 1:
            return 1
        # Simulate recursive call via the simulator
        return n * simulator.call("factorial", n - 1)

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    def cause_overflow() -> NoReturn:
        # Intentionally cause stack overflow
        simulator.call("cause_overflow")

    # Register functions
    simulator.register_function(factorial)
    simulator.register_function(greet)
    simulator.register_function(cause_overflow)

    # Simulate function calls
    try:
        print("Call Stack before any call:", simulator.stack_trace())
        print("greet('Alice'):", simulator.call("greet", "Alice"))
        print("factorial(5):", simulator.call("factorial", 5))
        print("Call Stack after calls:", simulator.stack_trace())
    except Exception as exc:
        print(f"Error: {exc}")

    # Demonstrate stack overflow handling
    try:
        simulator.call("cause_overflow")
    except CallStackOverflowError as exc:
        print(f"Stack Overflow Caught: {exc}")

    # Demonstrate underflow handling
    try:
        empty_stack = CallStack()
        empty_stack.pop()
    except CallStackUnderflowError as exc:
        print(f"Stack Underflow Caught: {exc}")

    # Demonstrate error handling for unregistered function
    try:
        simulator.call("non_existent_function")
    except KeyError as exc:
        print(f"Function Not Registered: {exc}")

    # Demonstrate error handling for invalid function argument
    try:
        simulator.call("factorial", -1)
    except ValueError as exc:
        print(f"Invalid Argument: {exc}")