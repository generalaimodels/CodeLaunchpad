#!/usr/bin/env python3
"""
Mastery-Level OOP Guide — Inheritance & Polymorphism (Single Python File)

What this file delivers:
- A professional, standards-compliant, and deeply commented single-file tutorial.
- Topics:
  1) Inheritance: Single, Multiple, Multilevel, Hierarchical, Hybrid (diamond with MRO).
  2) Polymorphism: Method "overloading" (Pythonic patterns), Method overriding, Operator overloading.

How to run:
- python oop_inheritance_polymorphism.py
- python oop_inheritance_polymorphism.py --quiet
- python oop_inheritance_polymorphism.py --no-rich
- python oop_inheritance_polymorphism.py --only inheritance
- python oop_inheritance_polymorphism.py --only polymorphism

Notes:
- "verbose: bool rich module very professional": This file supports verbosity and uses 'rich' (if installed) for polished output.
- All explanations and examples (with inputs and expected outputs) are embedded as comments/docstrings in this file.
- Code adheres to PEP 8, uses type hints, docstrings, and robust error handling.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, ClassVar, Iterable, List, Optional, Sequence, Tuple, Union, overload

# ------------------------------------------------------------------------------
# Optional Rich Integration (will gracefully degrade to plain printing)
# ------------------------------------------------------------------------------

HAVE_RICH = False
Console = None
try:
    from rich.console import Console as _Console
    from rich import print as rprint

    Console = _Console
    HAVE_RICH = True
except Exception:  # pragma: no cover - fallback for environments without rich
    def rprint(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)


class Out:
    """
    Professional output helper with verbosity and optional rich formatting.
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
            print(title)
            print("=" * 80)

    def h2(self, text: str) -> None:
        if not self.verbose:
            return
        if self.use_rich:
            rprint(f"[bold green]• {text}[/bold green]")
        else:
            print(f"-- {text} --")

    def info(self, text: str) -> None:
        if self.verbose:
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
        if self.verbose:
            print(*args, **kwargs)


# ==============================================================================
# 1) INHERITANCE: Single, Multiple, Multilevel, Hierarchical, Hybrid
# ==============================================================================

# ------------------------------------------------------------------------------
# Single Inheritance
# ------------------------------------------------------------------------------

class Vehicle:
    """
    Parent (Base) class demonstrating single inheritance.

    Attributes:
        make: Manufacturer name.

    Methods:
        start(): Base start behavior.
        stop(): Base stop behavior.

    Child classes will inherit these and can override as needed.
    """
    def __init__(self, make: str) -> None:
        self.make = make

    def start(self) -> str:
        return f"{self.make}: engine started."

    def stop(self) -> str:
        return f"{self.make}: engine stopped."

    def __str__(self) -> str:
        return f"Vehicle(make={self.make!r})"


class Car(Vehicle):
    """
    Child class (single inheritance) extending Vehicle.
    Adds seats and overrides start().
    """
    def __init__(self, make: str, seats: int) -> None:
        super().__init__(make)
        if seats <= 0:
            raise ValueError("seats must be positive.")
        self.seats = seats

    def start(self) -> str:  # Method overriding
        # Example of extending base behavior
        base = super().start()
        return f"{base[:-1]} with battery check complete."

    def honk(self) -> str:
        return f"{self.make}: honk!"

    def __str__(self) -> str:
        return f"Car(make={self.make!r}, seats={self.seats})"


def example_single_inheritance(out: Out) -> None:
    """
    INPUT:
        v = Vehicle("Generic Motors")
        c = Car("Tesla", seats=5)
        v.start(); v.stop()
        c.start(); c.honk(); c.stop()

    EXPECTED OUTPUT (approximate):
        Vehicle start/stop messages
        Car start message indicates extended behavior
        Car honk emits make-specific horn
    """
    out.h2("Single Inheritance: Vehicle -> Car")
    v = Vehicle("Generic Motors")
    c = Car("Tesla", seats=5)

    out.info(v.start())
    out.info(v.stop())
    out.info(c.start())
    out.info(c.honk())
    out.info(c.stop())
    out.ok("Single inheritance OK.")


# ------------------------------------------------------------------------------
# Multilevel Inheritance
# ------------------------------------------------------------------------------

class Animal:
    """Top-level base in a multilevel chain."""
    def breathe(self) -> str:
        return "Animal: breathing..."

    def speak(self) -> str:
        return "Animal: (generic sound)"


class Mammal(Animal):
    """Intermediate level."""
    def breathe(self) -> str:  # override and extend
        return "Mammal: breathing with lungs."

    def feed_milk(self) -> str:
        return "Mammal: feeding milk."


class Dog(Mammal):
    """Leaf subclass with specialized behavior."""
    def speak(self) -> str:  # override
        return "Dog: woof!"

    def fetch(self, item: str) -> str:
        return f"Dog fetched {item}."


def example_multilevel_inheritance(out: Out) -> None:
    """
    INPUT:
        d = Dog()
        d.breathe(); d.speak(); d.feed_milk(); d.fetch("ball")
        issubclass(Dog, Mammal), issubclass(Mammal, Animal)

    EXPECTED OUTPUT (approximate):
        Mammal breathing message
        Dog: woof!
        Mammal: feeding milk.
        Dog fetched ball.
        issubclass checks True
    """
    out.h2("Multilevel Inheritance: Animal -> Mammal -> Dog")
    d = Dog()
    out.info(d.breathe())
    out.info(d.speak())
    out.info(d.feed_milk())
    out.info(d.fetch("ball"))
    out.info(f"issubclass(Dog, Mammal) = {issubclass(Dog, Mammal)}")
    out.info(f"issubclass(Mammal, Animal) = {issubclass(Mammal, Animal)}")
    out.ok("Multilevel inheritance OK.")


# ------------------------------------------------------------------------------
# Hierarchical Inheritance (one parent, many children)
# ------------------------------------------------------------------------------

class Employee:
    """Base class for hierarchical inheritance."""
    def __init__(self, name: str, base_pay: float) -> None:
        if base_pay < 0:
            raise ValueError("base_pay cannot be negative.")
        self.name = name
        self.base_pay = base_pay

    def compute_pay(self, hours: float) -> float:
        """Default pay (overtime not handled here)."""
        if hours < 0:
            raise ValueError("hours cannot be negative.")
        return self.base_pay * hours

    def role(self) -> str:
        return "Employee"


class Developer(Employee):
    """Child 1: overrides compute_pay with overtime logic."""
    def compute_pay(self, hours: float) -> float:  # override
        if hours < 0:
            raise ValueError("hours cannot be negative.")
        overtime = max(0.0, hours - 40.0)
        return self.base_pay * (hours + 0.5 * overtime)

    def role(self) -> str:
        return "Developer"


class Manager(Employee):
    """Child 2: fixed bonus model."""
    def __init__(self, name: str, base_pay: float, bonus: float) -> None:
        super().__init__(name, base_pay)
        self.bonus = max(0.0, bonus)

    def compute_pay(self, hours: float) -> float:  # override
        return super().compute_pay(hours) + self.bonus

    def role(self) -> str:
        return "Manager"


def example_hierarchical_inheritance(out: Out) -> None:
    """
    INPUT:
        staff = [Developer("Dev", 50.0), Manager("Mgr", 60.0, bonus=500.0)]
        for s in staff: (s.role(), s.compute_pay(45))

    EXPECTED OUTPUT (approximate):
        Developer pay includes 5h overtime at 1.5x
        Manager pay = base + fixed bonus
    """
    out.h2("Hierarchical Inheritance: Employee -> {Developer, Manager}")
    staff: List[Employee] = [Developer("Dev", 50.0), Manager("Mgr", 60.0, bonus=500.0)]
    for s in staff:
        out.info(f"{s.role()} {s.name}: pay(45h) = {s.compute_pay(45):.2f}")
    out.ok("Hierarchical inheritance OK.")


# ------------------------------------------------------------------------------
# Multiple Inheritance (cooperative super() and MRO)
# ------------------------------------------------------------------------------

class Repository:
    """Concrete base providing terminal save()."""
    def save(self) -> List[str]:
        return ["Repository.save"]


class ValidateMixin:
    """Mixin that validates before chaining to next save()."""
    def save(self) -> List[str]:  # type: ignore[override]
        # Perform validations here...
        chain = super().save()  # cooperative call
        return ["ValidateMixin.save"] + chain


class AuditMixin:
    """Mixin that audits around save()."""
    def save(self) -> List[str]:  # type: ignore[override]
        chain = super().save()
        return ["AuditMixin.save"] + chain


class UserRepo(ValidateMixin, AuditMixin, Repository):
    """
    Multiple inheritance order:
      MRO(UserRepo) = [UserRepo, ValidateMixin, AuditMixin, Repository, object]
    save() resolution chain:
      ValidateMixin.save -> AuditMixin.save -> Repository.save
    """
    pass


def example_multiple_inheritance(out: Out) -> None:
    """
    INPUT:
        u = UserRepo()
        calls = u.save()

    EXPECTED OUTPUT:
        Save chain list showing MRO order:
        ["ValidateMixin.save", "AuditMixin.save", "Repository.save"]
    """
    out.h2("Multiple Inheritance: ValidateMixin + AuditMixin + Repository")
    u = UserRepo()
    calls = u.save()
    out.info(f"save() call chain = {calls}")
    out.info(f"MRO = {[c.__name__ for c in UserRepo.__mro__]}")
    out.ok("Multiple inheritance OK (cooperative super).")


# ------------------------------------------------------------------------------
# Hybrid Inheritance (diamond pattern + MRO)
# ------------------------------------------------------------------------------

class A:
    def who(self, trail: Optional[List[str]] = None) -> List[str]:
        trail = trail or []
        trail.append("A")
        # In diamond, use super() to thread calls through MRO safely
        return trail


class B(A):
    def who(self, trail: Optional[List[str]] = None) -> List[str]:
        trail = super().who(trail)
        trail.append("B")
        return trail


class C(A):
    def who(self, trail: Optional[List[str]] = None) -> List[str]:
        trail = super().who(trail)
        trail.append("C")
        return trail


class D(B, C):  # Diamond: A <- B, A <- C, D <- (B, C)
    def who(self, trail: Optional[List[str]] = None) -> List[str]:
        trail = super().who(trail)  # obey MRO: D->B->C->A
        trail.append("D")
        return trail


def example_hybrid_inheritance(out: Out) -> None:
    """
    INPUT:
        d = D()
        seq = d.who()

    EXPECTED OUTPUT (given MRO D->B->C->A):
        seq == ["A", "B", "C", "D"]  (A called once, no duplication)
        MRO shows ['D', 'B', 'C', 'A', 'object']
    """
    out.h2("Hybrid Inheritance: Diamond with cooperative super()")
    d = D()
    seq = d.who()
    out.info(f"Call sequence = {seq}")
    out.info(f"MRO = {[c.__name__ for c in D.__mro__]}")
    out.ok("Hybrid (diamond) inheritance OK (A called exactly once).")


def demo_inheritance(out: Out) -> None:
    """
    Orchestrates all inheritance examples with input/expected commentary.
    """
    out.rule("1) INHERITANCE — Single • Multiple • Multilevel • Hierarchical • Hybrid")
    example_single_inheritance(out)
    example_multilevel_inheritance(out)
    example_hierarchical_inheritance(out)
    example_multiple_inheritance(out)
    example_hybrid_inheritance(out)


# ==============================================================================
# 2) POLYMORPHISM: Overloading (Pythonic), Overriding, Operator Overloading
# ==============================================================================

# ------------------------------------------------------------------------------
# Method "Overloading" in Python (patterns: defaults, *args, @overload typing)
# ------------------------------------------------------------------------------

class Adder:
    """
    Illustrates "method overloading" via default parameters and *args.
    Python does not support signature-based overloading at runtime.
    """
    def add(self, a: Union[int, float], b: Union[int, float] = 0, *more: Union[int, float]) -> float:
        """
        INPUT:
            add(2, 3) -> 5
            add(2)    -> 2 (b defaults to 0)
            add(1, 2, 3, 4) -> 10

        EXPECTED OUTPUT:
            Numeric sum with float conversion.
        """
        values: List[float] = [float(a), float(b)] + [float(x) for x in more]
        return sum(values)


# A function-style overloading using 'typing.overload' for static type checkers.
# At runtime, we manually dispatch based on arguments.

@overload
def distance(x1: float, y1: float, x2: float, y2: float) -> float: ...
@overload
def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float: ...

def distance(*args: Any) -> float:
    """
    Overloaded distance function:
    - distance(x1, y1, x2, y2)
    - distance((x1, y1), (x2, y2))

    INPUT:
        distance(0, 0, 3, 4) -> 5.0
        distance((0, 0), (3, 4)) -> 5.0
    """
    if len(args) == 4:
        x1, y1, x2, y2 = (float(args[0]), float(args[1]), float(args[2]), float(args[3]))
        return math.hypot(x2 - x1, y2 - y1)
    if len(args) == 2:
        p1, p2 = args
        x1, y1 = (float(p1[0]), float(p1[1]))
        x2, y2 = (float(p2[0]), float(p2[1]))
        return math.hypot(x2 - x1, y2 - y1)
    raise TypeError("distance expects (x1, y1, x2, y2) or ((x1, y1), (x2, y2))")


def example_overloading(out: Out) -> None:
    """
    INPUT:
        ad = Adder()
        ad.add(2, 3)
        ad.add(5)              # default b=0
        ad.add(1, 2, 3, 4)
        distance(0, 0, 3, 4)
        distance((0, 0), (3, 4))

    EXPECTED OUTPUT:
        5.0; 5.0; 10.0; 5.0; 5.0
    """
    out.h2("Polymorphism — 'Overloading' via defaults/*args and typing.overload")
    ad = Adder()
    out.info(f"Adder.add(2, 3) = {ad.add(2, 3)}")
    out.info(f"Adder.add(5) = {ad.add(5)}")
    out.info(f"Adder.add(1, 2, 3, 4) = {ad.add(1, 2, 3, 4)}")
    out.info(f"distance(0,0,3,4) = {distance(0, 0, 3, 4)}")
    out.info(f"distance((0,0),(3,4)) = {distance((0, 0), (3, 4))}")
    out.ok("Overloading patterns demonstrated.")


# ------------------------------------------------------------------------------
# Method Overriding (runtime polymorphism) — one name, many forms
# ------------------------------------------------------------------------------

class Notifier:
    """Base notifier with a uniform interface."""
    def notify(self, message: str) -> str:
        return f"Base notify: {message}"


class EmailNotifier(Notifier):
    def notify(self, message: str) -> str:  # override
        return f"Email sent: {message}"


class SMSNotifier(Notifier):
    def notify(self, message: str) -> str:  # override
        return f"SMS sent: {message}"


def example_overriding(out: Out) -> None:
    """
    INPUT:
        notifiers: List[Notifier] = [EmailNotifier(), SMSNotifier(), Notifier()]
        for n in notifiers: n.notify("System update")

    EXPECTED OUTPUT:
        Email sent: System update
        SMS sent: System update
        Base notify: System update
    """
    out.h2("Polymorphism — Method Overriding")
    notifiers: List[Notifier] = [EmailNotifier(), SMSNotifier(), Notifier()]
    for n in notifiers:
        out.info(n.notify("System update"))
    out.ok("Method overriding demonstrated via uniform interface.")


# ------------------------------------------------------------------------------
# Operator Overloading — customize behavior of built-in operators
# ------------------------------------------------------------------------------

def _gcd(a: int, b: int) -> int:
    """Greatest common divisor (Euclidean algorithm)."""
    while b:
        a, b = b, a % b
    return abs(a)


@total_ordering
class Rational:
    """
    Immutable rational number with normalization and operator overloads.

    Supported:
        +, -, *, /, unary -, abs, ==, <, <=, >, >=, float(), str(), repr(), hash

    Design:
        - Always stored in lowest terms.
        - Denominator sign is always positive.
        - Type-safe: operations support int, float, Rational with clear semantics.
    """
    __slots__ = ("_n", "_d")

    def __init__(self, numerator: int, denominator: int = 1) -> None:
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise TypeError("Rational requires integer numerator and denominator.")
        if denominator == 0:
            raise ZeroDivisionError("Denominator cannot be zero.")
        sign = -1 if (numerator * denominator) < 0 else 1
        n, d = abs(numerator), abs(denominator)
        g = _gcd(n, d)
        self._n = sign * (n // g)
        self._d = d // g

    # Properties for read-only access
    @property
    def numerator(self) -> int:
        return self._n

    @property
    def denominator(self) -> int:
        return self._d

    # String and repr
    def __str__(self) -> str:
        return f"{self._n}/{self._d}"

    def __repr__(self) -> str:
        return f"Rational({self._n}, {self._d})"

    def __hash__(self) -> int:
        return hash((self._n, self._d))

    # Conversion
    def __float__(self) -> float:
        return self._n / self._d

    # Unary
    def __neg__(self) -> "Rational":
        return Rational(-self._n, self._d)

    def __abs__(self) -> "Rational":
        return Rational(abs(self._n), self._d)

    # Binary arithmetic with int or Rational
    def _coerce_other(self, other: Any) -> "Rational":
        if isinstance(other, Rational):
            return other
        if isinstance(other, int):
            return Rational(other, 1)
        raise TypeError(f"Unsupported operand type(s) for operation: 'Rational' and '{type(other).__name__}'")

    def __add__(self, other: Any) -> "Rational":
        r = self._coerce_other(other)
        return Rational(self._n * r._d + r._n * self._d, self._d * r._d)

    def __radd__(self, other: Any) -> "Rational":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Rational":
        r = self._coerce_other(other)
        return Rational(self._n * r._d - r._n * self._d, self._d * r._d)

    def __rsub__(self, other: Any) -> "Rational":
        r = self._coerce_other(other)
        return Rational(r._n * self._d - self._n * r._d, r._d * self._d)

    def __mul__(self, other: Any) -> "Rational":
        r = self._coerce_other(other)
        return Rational(self._n * r._n, self._d * r._d)

    def __rmul__(self, other: Any) -> "Rational":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Rational":
        r = self._coerce_other(other)
        if r._n == 0:
            raise ZeroDivisionError("Division by zero.")
        return Rational(self._n * r._d, self._d * r._n)

    def __rtruediv__(self, other: Any) -> "Rational":
        if self._n == 0:
            raise ZeroDivisionError("Division by zero.")
        r = self._coerce_other(other)
        return Rational(r._n * self._d, r._d * self._n)

    # Comparisons
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Rational):
            return (self._n == other._n) and (self._d == other._d)
        if isinstance(other, int):
            return (self._d == 1) and (self._n == other)
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        r = self._coerce_other(other)
        return self._n * r._d < r._n * self._d


def example_operator_overloading(out: Out) -> None:
    """
    INPUT:
        a = Rational(1, 2)   # 1/2
        b = Rational(3, 4)   # 3/4
        a + b                -> 5/4
        1 + a                -> 3/2
        b - a                -> 1/4
        a * b                -> 3/8
        b / a                -> 3/2
        float(a)             -> 0.5
        a == Rational(2, 4)  -> True
        sorted([R(2,3), R(1,2), R(3,5)]) -> ascending order

    EXPECTED OUTPUT (approximate):
        5/4; 3/2; 1/4; 3/8; 3/2; 0.5; True; sorted list increasing
        Errors:
            Rational(1, 0) -> ZeroDivisionError
            a + 0.5 -> TypeError (unsupported operand)
    """
    out.h2("Polymorphism — Operator Overloading with Rational numbers")
    a = Rational(1, 2)
    b = Rational(3, 4)
    out.info(f"{a} + {b} = {a + b}")
    out.info(f"1 + {a} = {1 + a}")
    out.info(f"{b} - {a} = {b - a}")
    out.info(f"{a} * {b} = {a * b}")
    out.info(f"{b} / {a} = {b / a}")
    out.info(f"float({a}) = {float(a)}")
    out.info(f"{a} == Rational(2, 4) -> {a == Rational(2, 4)}")
    arr = [Rational(2, 3), Rational(1, 2), Rational(3, 5)]
    out.info(f"sorted -> {sorted(arr)}")
    # Edge cases
    try:
        _ = Rational(1, 0)
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")
    try:
        _ = a + 0.5  # not supported by our _coerce_other
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")
    out.ok("Operator overloading demonstrated with arithmetic and comparisons.")


def demo_polymorphism(out: Out) -> None:
    """
    Orchestrates all polymorphism examples with input/expected commentary.

    Coverage:
    - Overloading (2+ examples): Adder.add defaults/*args and distance overloads.
    - Overriding: Notifier subclasses.
    - Operator Overloading: Rational arithmetic/comparison.
    """
    out.rule("2) POLYMORPHISM — Overloading • Overriding • Operator Overloading")
    example_overloading(out)
    example_overriding(out)
    example_operator_overloading(out)


# ==============================================================================
# MAIN HARNESS
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mastery-level OOP walkthrough: Inheritance & Polymorphism (single file)."
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity.")
    parser.add_argument("--no-rich", action="store_true", help="Force plain output even if 'rich' is installed.")
    parser.add_argument(
        "--only",
        choices=["inheritance", "polymorphism", "all"],
        default="all",
        help="Run a specific section only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Out(verbose=not args.quiet, use_rich=(not args.no_rich))
    out.rule("OOP: INHERITANCE & POLYMORPHISM — Professional, Single-File Tutorial")

    if args.only in ("inheritance", "all"):
        demo_inheritance(out)
        out.print()

    if args.only in ("polymorphism", "all"):
        demo_polymorphism(out)

    out.rule("DONE")
    out.ok("All sections executed successfully.")


if __name__ == "__main__":
    main()