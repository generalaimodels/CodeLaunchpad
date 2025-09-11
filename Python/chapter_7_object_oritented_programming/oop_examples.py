#!/usr/bin/env python3
"""
Mastery-Level Guide: Class, Object, Encapsulation, Abstraction (ABC) — Single Python File

This single file is a curated tutorial-quality codebase that:
- Demonstrates how to define and use Classes and Objects.
- Teaches Encapsulation techniques (public/protected/private, properties, validation).
- Explains Abstraction using Python’s abc (Abstract Base Class) module.
- Provides robust comments, edge-case coverage, exceptions, and clear expected outputs.

Execution:
- python mastery_oop.py
- python mastery_oop.py --quiet        # less verbose output
- python mastery_oop.py --no-rich      # force plain output even if 'rich' is installed
- python mastery_oop.py --only SECTION # run a specific section: class, encapsulation, abstraction, all

Sections:
1) Class and Object
2) Encapsulation
3) Abstraction with ABC

Note:
- This file is self-contained; all explanations, inputs, and expected outputs are embedded as comments.
- Code follows professional standards: type hints, docstrings, PEP 8 style, and clean structure.
"""

from __future__ import annotations

import argparse
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, List, Optional, Tuple, Union

# Optional rich formatting
HAVE_RICH = False
Console = None
Table = None
try:
    from rich.console import Console as _Console
    from rich.table import Table as _Table
    from rich import print as rprint

    Console = _Console
    Table = _Table
    HAVE_RICH = True
except Exception:
    def rprint(*args: Any, **kwargs: Any) -> None:  # fallback to print
        print(*args, **kwargs)


# ------------------------------------------------------------------------------
# Output Utilities (supports verbose printing and optional rich formatting)
# ------------------------------------------------------------------------------

class Out:
    """
    Thin wrapper around printing to support:
    - verbose vs quiet mode
    - rich vs plain formatting
    """
    def __init__(self, verbose: bool = True, use_rich: bool = True) -> None:
        self.verbose = verbose
        self.use_rich = use_rich and HAVE_RICH
        self.console = Console() if self.use_rich else None

    def rule(self, title: str) -> None:
        if not self.verbose:
            return
        if self.use_rich:
            self.console.rule(f"[bold cyan]{title}[/bold cyan]")
        else:
            print("=" * 80)
            print(title.upper())
            print("=" * 80)

    def h2(self, text: str) -> None:
        if not self.verbose:
            return
        if self.use_rich:
            rprint(f"[bold green]• {text}[/bold green]")
        else:
            print(f"-- {text} --")

    def info(self, text: str) -> None:
        if not self.verbose:
            return
        if self.use_rich:
            rprint(f"[white]{text}[/white]")
        else:
            print(text)

    def ok(self, text: str) -> None:
        if not self.verbose:
            return
        if self.use_rich:
            rprint(f"[green]✓ {text}[/green]")
        else:
            print(f"[OK] {text}")

    def warn(self, text: str) -> None:
        if not self.verbose:
            return
        if self.use_rich:
            rprint(f"[yellow]⚠ {text}[/yellow]")
        else:
            print(f"[WARN] {text}")

    def err(self, text: str) -> None:
        if not self.verbose:
            return
        if self.use_rich:
            rprint(f"[red]✗ {text}[/red]")
        else:
            print(f"[ERR] {text}")

    def print(self, *args: Any, **kwargs: Any) -> None:
        if not self.verbose:
            return
        print(*args, **kwargs)


# ------------------------------------------------------------------------------
# 1) CLASS & OBJECT
# ------------------------------------------------------------------------------

class Vector2D:
    """
    Class: Vector2D
    - Blueprint for 2D vectors (x, y), with attributes and methods.
    - Demonstrates:
        • Class attributes vs instance attributes
        • Instance methods, staticmethod, classmethod
        • dunder methods for readability (__repr__/__str__/__add__/__mul__)
        • Object instantiation and usage patterns

    Attributes:
        x (float): x-coordinate
        y (float): y-coordinate

    Class Attributes:
        DIMENSIONS (int): dimensionality of the vector space (=2)
        ZERO (Vector2D): a shared zero vector for convenience

    Methods:
        magnitude() -> float
        normalized() -> Vector2D
        __add__/__sub__/__mul__/__rmul__
        distance(a, b) -> float (staticmethod)
        from_polar(r, theta) -> Vector2D (classmethod)
    """
    DIMENSIONS: ClassVar[int] = 2  # class-level constant

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x: float = float(x)
        self.y: float = float(y)

    def __repr__(self) -> str:
        return f"Vector2D(x={self.x:.3f}, y={self.y:.3f})"

    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f})"

    # Equality for testing expectations
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Vector2D):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    # Vector addition
    def __add__(self, other: "Vector2D") -> "Vector2D":
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x + other.x, self.y + other.y)

    # Vector subtraction
    def __sub__(self, other: "Vector2D") -> "Vector2D":
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x - other.x, self.y - other.y)

    # Scalar multiplication (v * k)
    def __mul__(self, scalar: float) -> "Vector2D":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector2D(self.x * scalar, self.y * scalar)

    # Scalar multiplication (k * v)
    def __rmul__(self, scalar: float) -> "Vector2D":
        return self.__mul__(scalar)

    def magnitude(self) -> float:
        """Euclidean length: sqrt(x^2 + y^2)"""
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vector2D":
        """Return a unit vector in the same direction. Handles zero safely."""
        mag = self.magnitude()
        if mag == 0.0:
            # Return a copy of zero vector to avoid division by zero
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / mag, self.y / mag)

    @staticmethod
    def distance(a: "Vector2D", b: "Vector2D") -> float:
        """Static utility that doesn't depend on a specific instance."""
        return (a - b).magnitude()

    @classmethod
    def from_polar(cls, r: float, theta_radians: float) -> "Vector2D":
        """Construct vector from polar coordinates (r, θ)."""
        return cls(r * math.cos(theta_radians), r * math.sin(theta_radians))


# Class-level shared ZERO initialized after class is defined
Vector2D.ZERO: ClassVar[Vector2D] = Vector2D(0.0, 0.0)


def demo_class_and_object(out: Out) -> None:
    """
    Demonstrate Class vs Object with Vector2D.

    INPUT:
        v1 = Vector2D(3, 4)
        v2 = Vector2D.ZERO
        v3 = v1 + Vector2D(1, 2)
        mag_v1 = v1.magnitude()
        unit_v1 = v1.normalized()
        dist = Vector2D.distance(v1, v3)
        v_from_polar = Vector2D.from_polar(2, math.pi/2)

    EXPECTED OUTPUT (approximate):
        v1 = (3.000, 4.000)
        |v1| = 5.000
        v2 = (0.000, 0.000)
        v3 = (4.000, 6.000)
        v1 + v2 = (3.000, 4.000)
        2 * v1 = (6.000, 8.000)
        unit(v1) = (0.600, 0.800)
        distance(v1, v3) = 2.236...
        from_polar(2, pi/2) = (0.000, 2.000)
        DIMENSIONS = 2
    """
    out.rule("1) CLASS AND OBJECT")
    out.h2("Create objects from a class (instances) and use methods/attributes")

    v1 = Vector2D(3, 4)
    v2 = Vector2D.ZERO  # shared class-level zero vector
    v3 = v1 + Vector2D(1, 2)

    out.info(f"v1 = {v1}")
    out.info(f"|v1| = {v1.magnitude():.3f}")
    out.info(f"v2 = {v2}")
    out.info(f"v3 = {v3}")
    out.info(f"v1 + v2 = {v1 + v2}")
    out.info(f"2 * v1 = {2 * v1}")
    out.info(f"unit(v1) = {v1.normalized()}")

    dist = Vector2D.distance(v1, v3)
    out.info(f"distance(v1, v3) = {dist:.3f}")

    v_from_polar = Vector2D.from_polar(2.0, math.pi / 2)
    out.info(f"from_polar(2, pi/2) = {v_from_polar}")

    out.info(f"DIMENSIONS = {Vector2D.DIMENSIONS}")
    out.ok("Class/Object basics demonstrated successfully.")


# ------------------------------------------------------------------------------
# 2) ENCAPSULATION
# ------------------------------------------------------------------------------

class BankError(Exception):
    """Base exception for bank-related errors."""


class InvalidOperation(BankError):
    """Raised for invalid operations (e.g., wrong PIN)."""


class InsufficientFunds(BankError):
    """Raised when trying to withdraw more than the balance."""


class BankAccount:
    """
    Encapsulation Example: BankAccount

    Shows how to:
    - Wrap data and behavior in a single unit (class)
    - Restrict direct access using naming conventions and properties
    - Validate inputs in setters/methods
    - Use "protected" and "private" attributes by convention

    Public API (safe to use):
        .owner (str)
        .balance (float) — read-only property
        .deposit(amount: float) -> None
        .withdraw(amount: float, pin: str) -> float
        .set_pin(old_pin: Optional[str], new_pin: str) -> None
        .statement() -> str
        .apply_interest() -> None

    Protected attributes (convention, single underscore):
        ._transactions: List[str] — accessible to subclasses

    Private attributes (name-mangled, double underscore):
        .__pin: str
        .__account_number: str

    Class attributes:
        INTEREST_RATE (float): class-level shared interest rate
    """
    INTEREST_RATE: ClassVar[float] = 0.02  # 2% annually (for demo)

    def __init__(self, owner: str, opening_balance: float, pin: str, account_number: str) -> None:
        if opening_balance < 0:
            raise ValueError("Opening balance cannot be negative.")
        self.owner: str = owner
        self._balance: float = float(opening_balance)        # "protected" by convention
        self._transactions: List[str] = []                   # "protected" - for subclass usage
        self.__pin: str = self._validate_pin(pin)            # "private" via name-mangling
        self.__account_number: str = self._validate_account_number(account_number)  # private

        self._log(f"ACCOUNT OPENED: Owner={self.owner}, Balance={self._balance:.2f}")

    # ---------------------- Encapsulated helpers and validators ----------------------

    @staticmethod
    def _validate_amount(amount: float) -> float:
        if not isinstance(amount, (int, float)):
            raise TypeError("Amount must be numeric.")
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        return float(amount)

    @staticmethod
    def _validate_pin(pin: str) -> str:
        if not isinstance(pin, str) or not pin.isdigit() or len(pin) not in (4, 6):
            raise ValueError("PIN must be a string of 4 or 6 digits.")
        return pin

    @staticmethod
    def _validate_account_number(acc: str) -> str:
        if not isinstance(acc, str) or not acc.isdigit() or len(acc) != 10:
            raise ValueError("Account number must be a 10-digit string.")
        return acc

    def _check_pin(self, pin: str) -> None:
        # Private credential check
        if pin != self.__pin:
            raise InvalidOperation("Invalid PIN.")

    def _log(self, entry: str) -> None:
        # Internal transaction log (protected by convention)
        self._transactions.append(entry)

    # ---------------------- Public API with properties & methods ----------------------

    @property
    def balance(self) -> float:
        """Read-only property for balance; no direct external setting."""
        return self._balance

    def deposit(self, amount: float) -> None:
        amount = self._validate_amount(amount)
        self._balance += amount
        self._log(f"DEPOSIT: +{amount:.2f}, New Balance={self._balance:.2f}")

    def withdraw(self, amount: float, pin: str) -> float:
        amount = self._validate_amount(amount)
        self._check_pin(pin)
        if amount > self._balance:
            raise InsufficientFunds(f"Insufficient funds to withdraw {amount:.2f}.")
        self._balance -= amount
        self._log(f"WITHDRAW: -{amount:.2f}, New Balance={self._balance:.2f}")
        return amount

    def set_pin(self, old_pin: Optional[str], new_pin: str) -> None:
        # Allow PIN set on new accounts without old_pin if transactions are zero
        if self._transactions and old_pin is None:
            raise InvalidOperation("Old PIN required to change PIN.")
        if old_pin is not None:
            self._check_pin(old_pin)
        self.__pin = self._validate_pin(new_pin)
        self._log("PIN CHANGED")

    def statement(self) -> str:
        # Sensitive data (account number) is not exposed publicly
        # Last 4 digits only for display
        masked = f"****{self.__account_number[-4:]}"
        lines = [
            f"Statement for {self.owner} [{masked}]",
            f"Balance: {self._balance:.2f}",
            "Activity:"
        ] + [f" - {t}" for t in self._transactions]
        return "\n".join(lines)

    def apply_interest(self) -> None:
        """Apply class-level interest rate to the balance."""
        earned = self._balance * self.INTEREST_RATE
        self._balance += earned
        self._log(f"INTEREST: +{earned:.2f}, New Balance={self._balance:.2f}")

    # ---------------------- Class-level configuration ----------------------

    @classmethod
    def set_interest_rate(cls, new_rate: float) -> None:
        if not (0.0 <= new_rate <= 1.0):
            raise ValueError("Interest rate must be between 0.0 and 1.0.")
        cls.INTEREST_RATE = float(new_rate)


class AuditedBankAccount(BankAccount):
    """
    Subclass leveraging "protected" attribute _transactions to add audit hooks.
    Demonstrates that subclasses are allowed to use single-underscore attributes.
    """

    def deposit(self, amount: float) -> None:
        self._log("[AUDIT] deposit initiated")
        super().deposit(amount)
        self._log("[AUDIT] deposit completed")

    def withdraw(self, amount: float, pin: str) -> float:
        self._log("[AUDIT] withdraw initiated")
        amt = super().withdraw(amount, pin)
        self._log("[AUDIT] withdraw completed")
        return amt


def demo_encapsulation(out: Out) -> None:
    """
    Demonstrate Encapsulation with BankAccount.

    INPUT:
        acct = BankAccount("Alice", 100.0, pin="1234", account_number="1234567890")
        acct.deposit(50.0)
        acct.withdraw(80.0, pin="1234")
        acct.apply_interest()
        print(acct.balance)
        print(acct.statement())
        acct.set_pin(old_pin="1234", new_pin="4321")

        # Edge cases & exceptions:
        acct.deposit(-10)                   -> ValueError
        acct.withdraw(1e9, pin="4321")      -> InsufficientFunds
        acct.withdraw(10, pin="9999")       -> InvalidOperation
        acct.__pin                          -> AttributeError (name-mangled)
        acct._BankAccount__pin              -> Works (not recommended)

    EXPECTED OUTPUT (abridged):
        Balance after operations: 73.400  (exact depends on interest rate)
        Statement shows DEPOSIT, WITHDRAW, INTEREST entries
        PIN changed successfully
        Error: Amount must be positive.
        Error: Insufficient funds to withdraw ...
        Error: Invalid PIN.
        Error: 'BankAccount' object has no attribute '__pin'
        Direct access via name-mangling reveals PIN (for demonstration only)
    """
    out.rule("2) ENCAPSULATION")
    out.h2("Create account and use only the public API")

    acct = BankAccount("Alice", 100.0, pin="1234", account_number="1234567890")
    acct.deposit(50.0)
    withdrew = acct.withdraw(80.0, pin="1234")
    out.info(f"Withdrew: {withdrew:.2f}")

    # Apply interest (default 2%)
    acct.apply_interest()
    out.info(f"Balance (after interest): {acct.balance:.3f}")
    out.print()
    out.info("Statement:")
    out.print(acct.statement())

    out.h2("Change PIN through a controlled method")
    acct.set_pin(old_pin="1234", new_pin="4321")
    out.ok("PIN changed successfully.")

    out.h2("Edge cases and exceptions")
    # Negative amount
    try:
        acct.deposit(-10.0)
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    # Insufficient funds
    try:
        acct.withdraw(1e9, pin="4321")
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    # Invalid PIN
    try:
        acct.withdraw(10.0, pin="9999")
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    # Private attribute access attempt
    try:
        # This will raise AttributeError because __pin is name-mangled
        _ = acct.__pin  # type: ignore[attr-defined]
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    # Name-mangling demonstration (not recommended)
    out.warn("Name-mangling access (for demonstration only; do not do this in real code):")
    try:
        # Accessing private via mangled name; strongly discouraged outside debugging
        pin_value = getattr(acct, "_BankAccount__pin")
        out.info(f"Accessed mangled private __pin = {pin_value!r}")
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    out.h2("Subclass uses protected attributes")
    aud = AuditedBankAccount("Bob", 250.0, pin="2468", account_number="0001112223")
    aud.deposit(30.0)
    aud.withdraw(20.0, pin="2468")
    out.info("Audited account statement")
    out.print(aud.statement())

    out.ok("Encapsulation principles demonstrated with validation and access control.")


# ------------------------------------------------------------------------------
# 3) ABSTRACTION with ABC (Abstract Base Classes)
# ------------------------------------------------------------------------------

class Shape(ABC):
    """
    Abstract Base Class: Shape
    - Abstraction hides implementation details: users rely on the interface:
        • area() -> float
        • perimeter() -> float
        • name (property)
    - Concrete subclasses must implement these abstract members.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-friendly shape name."""

    @abstractmethod
    def area(self) -> float:
        """Compute area."""

    @abstractmethod
    def perimeter(self) -> float:
        """Compute perimeter."""

    def describe(self) -> str:
        """Concrete method using the abstract interface."""
        return f"{self.name}: area={self.area():.3f}, perimeter={self.perimeter():.3f}"


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = self._validate_radius(radius)

    @staticmethod
    def _validate_radius(r: float) -> float:
        if not isinstance(r, (int, float)):
            raise TypeError("Radius must be numeric.")
        if r <= 0:
            raise ValueError("Radius must be positive.")
        return float(r)

    @property
    def name(self) -> str:
        return "Circle"

    def area(self) -> float:
        return math.pi * self.radius**2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = self._validate_side(width, "width")
        self.height = self._validate_side(height, "height")

    @staticmethod
    def _validate_side(value: float, label: str) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be numeric.")
        if value <= 0:
            raise ValueError(f"{label} must be positive.")
        return float(value)

    @property
    def name(self) -> str:
        return "Rectangle"

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


class PaymentProcessor(ABC):
    """
    Interface-like ABC for processing payments.
    Demonstrates abstraction of essential features without exposing internals.
    """

    @abstractmethod
    def process_payment(self, amount: float) -> str:
        """Process a payment and return a transaction id."""

    @abstractmethod
    def refund(self, amount: float) -> str:
        """Issue a refund and return a reference."""


class StripeProcessor(PaymentProcessor):
    def __init__(self, api_key: str) -> None:
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string.")
        self._api_key = api_key  # encapsulated; not exposed

    def process_payment(self, amount: float) -> str:
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        # Implementation detail hidden; pretend we call Stripe here
        return f"stripe_txn_{abs(hash((self._api_key, amount))) % 1_000_000}"

    def refund(self, amount: float) -> str:
        if amount <= 0:
            raise ValueError("Refund amount must be positive.")
        # Implementation detail hidden; pretend refund issued
        return f"stripe_ref_{abs(hash((self._api_key, amount, 'refund'))) % 1_000_000}"


def checkout(processor: PaymentProcessor, amount: float) -> str:
    """
    Client code that depends only on the abstraction (PaymentProcessor).
    Polymorphism: any concrete implementation will work as long as it respects the interface.
    """
    txn_id = processor.process_payment(amount)
    return f"Payment successful: {txn_id}"


def demo_abstraction(out: Out) -> None:
    """
    Demonstrate Abstraction with ABCs: Shape and PaymentProcessor.

    INPUT:
        # Shape ABC usage
        try: Shape()                       -> TypeError (cannot instantiate ABC)
        c = Circle(1.5); r = Rectangle(2, 3)
        [s.describe() for s in (c, r)]

        # PaymentProcessor "interface"
        gateway = StripeProcessor(api_key="sk_test_123")
        checkout(gateway, 49.99)
        gateway.refund(10.00)

        # Edge cases
        Circle(-1)                         -> ValueError
        Rectangle(0, 2)                    -> ValueError
        StripeProcessor("")                -> ValueError

    EXPECTED OUTPUT (abridged):
        Error: Can't instantiate abstract class Shape ...
        Circle: area=7.069, perimeter=9.425
        Rectangle: area=6.000, perimeter=10.000
        Payment successful: stripe_txn_<id>
        Refund id: stripe_ref_<id>
    """
    out.rule("3) ABSTRACTION WITH ABC")
    out.h2("Shapes: abstract interface with concrete implementations")
    try:
        _ = Shape()  # type: ignore[abstract]
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    shapes: List[Shape] = [Circle(1.5), Rectangle(2, 3)]
    for s in shapes:
        out.info(s.describe())

    out.h2("PaymentProcessor: interface-like ABC")
    gateway = StripeProcessor(api_key="sk_test_123")
    out.info(checkout(gateway, 49.99))
    ref = gateway.refund(10.00)
    out.info(f"Refund successful: {ref}")

    out.h2("Edge cases & validation")
    for bad in (
        ("Circle radius -1", lambda: Circle(-1)),
        ("Rectangle width 0", lambda: Rectangle(0, 2)),
        ("Empty API key", lambda: StripeProcessor("")),
    ):
        label, fn = bad
        try:
            fn()
        except Exception as e:
            out.err(f"{label} -> {type(e).__name__}: {e}")

    out.ok("Abstraction: users rely on essential features; details remain hidden.")


# ------------------------------------------------------------------------------
# MAIN: Orchestrate Demos
# ------------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mastery-level, single-file tutorial on Class, Object, Encapsulation, Abstraction (ABC)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity of output.",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Force plain output (no rich), even if 'rich' is installed.",
    )
    parser.add_argument(
        "--only",
        choices=["class", "encapsulation", "abstraction", "all"],
        default="all",
        help="Run a specific section.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    out = Out(verbose=not args.quiet, use_rich=(not args.no_rich))

    # Summary header
    out.rule("OBJECT-ORIENTED PROGRAMMING: CLASS • OBJECT • ENCAPSULATION • ABSTRACTION")

    if args.only in ("class", "all"):
        demo_class_and_object(out)
        out.print()

    if args.only in ("encapsulation", "all"):
        demo_encapsulation(out)
        out.print()

    if args.only in ("abstraction", "all"):
        demo_abstraction(out)

    if out.verbose:
        out.rule("DONE")
        out.ok("All sections executed.")


if __name__ == "__main__":
    main()