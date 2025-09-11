#!/usr/bin/env python3
"""
Advanced OOP in Python — Single-File Mastery (with rich-ready verbose output)

This single Python file is a precision-crafted, tutorial-grade codebase covering:
- Constructor (__init__)
- Destructor (__del__) + safe alternatives (weakref.finalize)
- self keyword (instance context)
- Class Method (@classmethod)
- Static Method (@staticmethod)
- Properties & Getters/Setters (@property)
- Duck Typing (behavior over types)
- Magic/Dunder Methods (__str__, __len__, __eq__, __add__, __call__, etc.)
- Abstract Base Classes (ABC)
- Metaclasses (class factories)

How to run:
- python advanced_oop.py
- python advanced_oop.py --quiet           # less verbose
- python advanced_oop.py --no-rich         # no color even if rich installed
- python advanced_oop.py --only constructor  # run a specific topic
  choices: constructor, destructor, self, classmethod, staticmethod, properties, duck, dunder, abc, metaclass, all

Note:
- "verbose: bool rich module very professional": honored via CLI flags.
- All explanations are in-code via docstrings/comments. Inputs and expected outputs are documented per demo.
- Code adheres to modern standards: type hints, PEP 8 naming, clean structure, robust validation.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import tempfile
import weakref
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

# ------------------------------------------------------------------------------
# Optional Rich Integration (polished output if available; graceful fallback)
# ------------------------------------------------------------------------------

HAVE_RICH = False
Console = None
try:
    from rich.console import Console as _Console
    from rich import print as rprint

    Console = _Console
    HAVE_RICH = True
except Exception:  # pragma: no cover
    def rprint(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)


class Out:
    """
    Output helper: verbose gating + optional rich styling.
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
# 1) CONSTRUCTOR (__init__)
# ==============================================================================

class Account:
    """
    Demonstrates __init__ with validation and computed fields.

    Behavior:
    - Validates username/email types and formats.
    - Computes a slug once at initialization.
    - Tracks instances via a class-level counter.

    Example 1 (valid):
        INPUT:
            a = Account("Ada", "ada@example.com")
            print(a.username, a.email, a.slug, Account.instances())
        EXPECTED:
            Ada ada@example.com ada Account.instances() >= 1

    Example 2 (invalid email):
        INPUT:
            Account("Bob", "not-an-email")
        EXPECTED:
            ValueError: invalid email format
    """
    _count: int = 0  # class-level instance counter

    def __init__(self, username: str, email: str) -> None:
        if not isinstance(username, str) or not username:
            raise TypeError("username must be a non-empty string.")
        if not isinstance(email, str) or "@" not in email or email.count("@") != 1:
            raise ValueError("invalid email format")
        self.username: str = username
        self.email: str = email
        # A computed attribute derived during construction (immutable by intent)
        self.slug: str = self.username.lower().replace(" ", "-")
        Account._count += 1

    @classmethod
    def instances(cls) -> int:
        return cls._count


class Matrix:
    """
    __init__ with structural validation and normalization.

    Rules:
    - Data must be a non-empty rectangular list of lists of numbers.
    - Internally normalizes all elements to float.

    Example 1:
        INPUT:
            m = Matrix([[1, 2], [3, 4.5]])
            m.shape -> (2, 2)
        EXPECTED:
            shape (2, 2); elements as floats

    Example 2 (ragged rows):
        INPUT:
            Matrix([[1, 2], [3]])
        EXPECTED:
            ValueError: rows must have equal length
    """
    def __init__(self, data: List[List[Union[int, float]]]) -> None:
        if not isinstance(data, list) or not data or not all(isinstance(r, list) and r for r in data):
            raise ValueError("data must be a non-empty list of non-empty lists.")
        row_len = len(data[0])
        if any(len(r) != row_len for r in data):
            raise ValueError("rows must have equal length.")
        self._data: List[List[float]] = [[float(x) for x in row] for row in data]
        self.rows: int = len(self._data)
        self.cols: int = row_len

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)


def demo_constructor(out: Out) -> None:
    out.rule("Constructor (__init__)")
    # Example 1
    out.h2("Example 1: Account construction with validation and computed fields")
    a = Account("Ada Lovelace", "ada@example.com")
    out.info(f"username={a.username}, email={a.email}, slug={a.slug}, instances={Account.instances()}")
    # Example 2
    out.h2("Example 2: Matrix validation (rectangular shape enforced)")
    m = Matrix([[1, 2], [3, 4.5]])
    out.info(f"Matrix shape = {m.shape}")
    try:
        _ = Matrix([[1, 2], [3]])  # ragged
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    try:
        _ = Account("", "x@y.com")
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    try:
        _ = Account("User", "not-email")
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")
    out.ok("__init__ validated inputs and created objects correctly.")


# ==============================================================================
# 2) DESTRUCTOR (__del__) — with caveats and safe alternative
# ==============================================================================

class TempFileResource:
    """
    Demonstrates __del__ for resource cleanup (NOT ALWAYS RELIABLE).

    Notes:
    - __del__ timing is non-deterministic (esp. across interpreters).
    - Prefer context managers or weakref.finalize for guaranteed cleanup.
    - Here we use __del__ to close and delete a temporary file.

    Example 1:
        INPUT:
            path = None
            def scope():
                r = TempFileResource()
                r.write("hello")
                return r.path
            path = scope()
            # object goes out of scope; we force collection:
            gc.collect()
        EXPECTED (printed by __del__):
            [__del__] Closed and removed temp file: <path>
    """
    def __init__(self) -> None:
        f = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8")
        self.path: str = f.name
        self._f = f

    def write(self, text: str) -> None:
        self._f.write(text)
        self._f.flush()

    def __del__(self) -> None:
        # DO NOT raise inside __del__; always guard cleanup
        try:
            if getattr(self, "_f", None):
                try:
                    self._f.close()
                except Exception:
                    pass
            if getattr(self, "path", None) and os.path.exists(self.path):
                try:
                    os.remove(self.path)
                except Exception:
                    pass
            # Use plain print; Out may be unavailable during interpreter shutdown
            print(f"[__del__] Closed and removed temp file: {getattr(self, 'path', None)}")
        except Exception:
            # Suppress all; exceptions in __del__ are ignored but may emit warnings
            pass


class FinalizedResource:
    """
    Safer alternative to __del__ using weakref.finalize.

    Example 2:
        INPUT:
            messages = []
            def scope():
                r = FinalizedResource(lambda p: messages.append(f'finalized:{p}'))
                r.touch()
            scope(); gc.collect()
        EXPECTED:
            messages contains one entry with 'finalized:<path>'
    """
    def __init__(self, on_finalize: Callable[[str], None]) -> None:
        f = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8")
        self.path: str = f.name
        self._f = f
        # finalize registers a callback independent of __del__
        self._finalizer = weakref.finalize(self, self._cleanup, on_finalize, self.path, self._f)

    @staticmethod
    def _cleanup(cb: Callable[[str], None], path: str, f) -> None:
        try:
            try:
                f.close()
            except Exception:
                pass
            if os.path.exists(path):
                os.remove(path)
        finally:
            cb(path)

    def touch(self) -> None:
        self._f.write("x")
        self._f.flush()


def demo_destructor(out: Out) -> None:
    out.rule("Destructor (__del__) and Safe Finalization")
    out.h2("Example 1: __del__ cleans up a temp file (timing may vary)")
    path_holder: List[str] = []

    def scope1() -> None:
        r = TempFileResource()
        r.write("hello")
        path_holder.append(r.path)

    scope1()
    # Encourage finalization (works reliably in CPython, but not guaranteed by spec)
    gc.collect()
    out.info(f"Temp file previously at: {path_holder[-1]} (should be removed)")

    out.h2("Example 2: weakref.finalize (robust cleanup, even with cycles)")
    msgs: List[str] = []

    def scope2() -> None:
        r = FinalizedResource(lambda p: msgs.append(f"finalized:{p}"))
        r.touch()
        # Create a cycle to test: r -> list -> r (still finalized on GC)
        ring: List[Any] = [r]
        ring.append(ring)

    scope2()
    gc.collect()
    out.info(f"Finalize callbacks captured: {msgs}")
    out.ok("Destructor concepts illustrated; prefer finalize/context managers for safety.")


# ==============================================================================
# 3) self Keyword — current instance reference
# ==============================================================================

class QueryBuilder:
    """
    Fluent interface using self (method chaining).

    Example 1:
        INPUT:
            q = QueryBuilder().select("id", "name").where("age > 18").order_by("name")
            q.build()
        EXPECTED:
            'SELECT id, name FROM table WHERE age > 18 ORDER BY name'

    Example 2:
        INPUT:
            QueryBuilder().where("active=1").select("*").build()
        EXPECTED (order independent due to builder use):
            'SELECT * FROM table WHERE active=1'
    """
    def __init__(self) -> None:
        self._columns: List[str] = []
        self._where: Optional[str] = None
        self._order_by: Optional[str] = None

    def select(self, *columns: str) -> "QueryBuilder":
        self._columns = list(columns) if columns else ["*"]
        return self  # method chaining returns self

    def where(self, clause: str) -> "QueryBuilder":
        self._where = clause
        return self

    def order_by(self, key: str) -> "QueryBuilder":
        self._order_by = key
        return self

    def build(self) -> str:
        cols = ", ".join(self._columns or ["*"])
        sql = f"SELECT {cols} FROM table"
        if self._where:
            sql += f" WHERE {self._where}"
        if self._order_by:
            sql += f" ORDER BY {self._order_by}"
        return sql


class SelfVsClass:
    """
    Distinguish instance state (self) vs class state (cls).

    Example 1:
        INPUT:
            a = SelfVsClass(); b = SelfVsClass()
            a.inc(); a.inc(); b.inc()
            (a.value, b.value, SelfVsClass.total)
        EXPECTED:
            (2, 1, 3)  # per-instance vs class aggregate

    Example 2:
        INPUT:
            SelfVsClass.total
            a.reset(); b.reset(); SelfVsClass.total
        EXPECTED:
            total reduced to 0
    """
    total: int = 0  # class variable (shared)
    def __init__(self) -> None:
        self.value: int = 0  # instance variable (per object)

    def inc(self) -> None:
        self.value += 1
        type(self).total += 1  # or SelfVsClass.total

    def reset(self) -> None:
        type(self).total -= self.value
        self.value = 0


def demo_self(out: Out) -> None:
    out.rule("self Keyword (instance context)")
    out.h2("Example 1: Fluent builder returns self")
    q = QueryBuilder().select("id", "name").where("age > 18").order_by("name")
    out.info(q.build())

    out.h2("Example 2: self vs class state")
    a = SelfVsClass()
    b = SelfVsClass()
    a.inc(); a.inc(); b.inc()
    out.info(f"a.value={a.value}, b.value={b.value}, total={SelfVsClass.total}")
    a.reset(); b.reset()
    out.info(f"After reset: a.value={a.value}, b.value={b.value}, total={SelfVsClass.total}")
    out.ok("self usage and class state interactions demonstrated.")


# ==============================================================================
# 4) Class Method (@classmethod)
# ==============================================================================

class User:
    """
    @classmethod as:
    - Alternative constructors
    - Class-wide configuration

    Example 1 (alt constructor):
        INPUT:
            u = User.from_fullname("Grace Hopper", email="g@navy.mil")
            (u.first, u.last, u.email)
        EXPECTED:
            ("Grace", "Hopper", "g@navy.mil")

    Example 2 (class config):
        INPUT:
            User.set_domain("example.org")
            u2 = User("Ada", "Lovelace")
            u2.email
        EXPECTED:
            "ada.lovelace@example.org"
    """
    _domain: str = "example.com"

    def __init__(self, first: str, last: str, email: Optional[str] = None) -> None:
        self.first = first
        self.last = last
        self.email = email or f"{first}.{last}@{self._domain}".lower()

    @classmethod
    def set_domain(cls, domain: str) -> None:
        if "." not in domain:
            raise ValueError("domain must contain a dot, e.g., 'example.org'")
        cls._domain = domain

    @classmethod
    def from_fullname(cls, fullname: str, email: Optional[str] = None) -> "User":
        first, _, last = fullname.partition(" ")
        if not last:
            raise ValueError("fullname must contain first and last name.")
        return cls(first, last, email=email)


class ShapeFactory:
    """
    @classmethod dispatch to create shapes by kind (shows polymorphic constructors).

    Example 1:
        INPUT:
            c = ShapeFactory.create("circle", radius=2.0)
            type(c).__name__, round(c.area(), 2)
        EXPECTED:
            ("Circle", 12.57)

    Example 2:
        INPUT:
            r = ShapeFactory.create("rectangle", width=2, height=3)
            round(r.perimeter(), 2)
        EXPECTED:
            10.0
    """
    @classmethod
    def create(cls, kind: str, **kwargs: Any) -> "IShape":
        kind = kind.lower()
        if kind == "circle":
            return Circle(kwargs["radius"])
        if kind == "rectangle":
            return Rectangle(kwargs["width"], kwargs["height"])
        raise ValueError(f"Unknown shape kind: {kind}")


def demo_classmethod(out: Out) -> None:
    out.rule("Class Method (@classmethod)")
    out.h2("Example 1: Alternative constructor from_fullname")
    u = User.from_fullname("Grace Hopper", email="g@navy.mil")
    out.info(f"{u.first=} {u.last=} {u.email=}")

    out.h2("Example 2: Class-wide configuration via @classmethod")
    User.set_domain("example.org")
    u2 = User("Ada", "Lovelace")
    out.info(f"Generated email with new domain: {u2.email}")

    try:
        User.set_domain("invalid-domain")
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    try:
        _ = User.from_fullname("Plato")  # missing last name
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    out.ok("@classmethod used for alt constructors and config.")


# ==============================================================================
# 5) Static Method (@staticmethod)
# ==============================================================================

class MathUtils:
    """
    Pure utilities grouped on a class via @staticmethod.

    Example 1:
        INPUT:
            MathUtils.is_prime(13), MathUtils.is_prime(1)
        EXPECTED:
            (True, False)

    Example 2:
        INPUT:
            MathUtils.hypot(3, 4)
        EXPECTED:
            5.0
    """
    @staticmethod
    def is_prime(n: int) -> bool:
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def hypot(x: float, y: float) -> float:
        return math.hypot(x, y)


class EmailValidator:
    """
    Validator methods as static (no instance/class state).

    Example 1:
        INPUT:
            EmailValidator.is_valid("a@b.com")
        EXPECTED:
            True

    Example 2:
        INPUT:
            EmailValidator.is_valid("no-at-symbol")
        EXPECTED:
            False
    """
    @staticmethod
    def is_valid(addr: str) -> bool:
        return isinstance(addr, str) and "@" in addr and "." in addr.split("@")[-1] and " " not in addr


def demo_staticmethod(out: Out) -> None:
    out.rule("Static Method (@staticmethod)")
    out.h2("Example 1: MathUtils prime test")
    out.info(f"is_prime(13) = {MathUtils.is_prime(13)}; is_prime(1) = {MathUtils.is_prime(1)}")
    out.h2("Example 2: Hypotenuse")
    out.info(f"hypot(3, 4) = {MathUtils.hypot(3, 4)}")

    out.h2("Email validation without class/instance state")
    out.info(f"EmailValidator.is_valid('a@b.com') = {EmailValidator.is_valid('a@b.com')}")
    out.info(f"EmailValidator.is_valid('no-at-symbol') = {EmailValidator.is_valid('no-at-symbol')}")
    out.ok("@staticmethod groups pure behavior logically under a class.")


# ==============================================================================
# 6) Properties & Getters/Setters (@property)
# ==============================================================================

class Temperature:
    """
    Encapsulation with @property: read/write with validation; computed property.

    Rules:
    - celsius must be >= absolute zero (-273.15)
    - fahrenheit is computed; setter updates celsius accordingly

    Example 1:
        INPUT:
            t = Temperature(0)
            (t.celsius, t.fahrenheit)
        EXPECTED:
            (0.0, 32.0)

    Example 2 (setter validation):
        INPUT:
            t.celsius = -300
        EXPECTED:
            ValueError (below absolute zero)
    """
    def __init__(self, celsius: float) -> None:
        self._celsius: float = 0.0
        self.celsius = celsius  # use setter for validation

    @property
    def celsius(self) -> float:
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        value = float(value)
        if value < -273.15:
            raise ValueError("Temperature below absolute zero.")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, f: float) -> None:
        self.celsius = (float(f) - 32) * 5 / 9  # delegate to validated setter


class Product:
    """
    Property with controlled access and derived attribute.

    Example 1:
        INPUT:
            p = Product("GPU", price=999.995, discount=0.10)
            (p.price, p.final_price)
        EXPECTED:
            price rounded to 2 decimals; final_price = price * (1 - discount)

    Example 2 (invalid discount):
        INPUT:
            p.discount = 1.5
        EXPECTED:
            ValueError: discount must be between 0 and 1
    """
    def __init__(self, name: str, price: float, discount: float = 0.0) -> None:
        self.name = name
        self._price: float = 0.0
        self._discount: float = 0.0
        self.price = price
        self.discount = discount

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, value: float) -> None:
        value = round(float(value), 2)
        if value < 0:
            raise ValueError("price cannot be negative.")
        self._price = value

    @property
    def discount(self) -> float:
        return self._discount

    @discount.setter
    def discount(self, value: float) -> None:
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError("discount must be between 0 and 1.")
        self._discount = value

    @property
    def final_price(self) -> float:
        return round(self._price * (1.0 - self._discount), 2)


def demo_properties(out: Out) -> None:
    out.rule("Properties & Getters/Setters (@property)")
    out.h2("Example 1: Temperature with computed Fahrenheit and validation")
    t = Temperature(0)
    out.info(f"celsius={t.celsius}, fahrenheit={t.fahrenheit}")
    t.fahrenheit = 212
    out.info(f"after setting fahrenheit=212 -> celsius={t.celsius:.2f}")

    out.h2("Example 2: Product price/discount controls")
    p = Product("GPU", 999.995, discount=0.10)
    out.info(f"price={p.price}, discount={p.discount}, final_price={p.final_price}")
    try:
        p.discount = 1.5
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")
    try:
        p.price = -1
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")
    out.ok("Properties encapsulate validation and expose clean API.")


# ==============================================================================
# 7) Duck Typing — "If it quacks like a duck..."
# ==============================================================================

class IShape(Protocol):
    """
    Structural typing aid (optional for static checkers, at runtime it's duck typing):
    Any object with area() -> float and perimeter() -> float is treated as a shape.
    """
    def area(self) -> float: ...
    def perimeter(self) -> float: ...


class Circle:
    def __init__(self, radius: float) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive.")
        self.radius = float(radius)

    def area(self) -> float:
        return math.pi * self.radius**2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

    def __str__(self) -> str:
        return f"Circle(r={self.radius})"


class Rectangle:
    def __init__(self, width: float, height: float) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height must be positive.")
        self.width = float(width)
        self.height = float(height)

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __str__(self) -> str:
        return f"Rectangle({self.width}x{self.height})"


class ThirdPartyDisk:
    """
    No inheritance from our classes; still compatible via duck typing because it implements .area().
    """
    def __init__(self, r: float) -> None:
        self.r = r

    def area(self) -> float:
        return math.pi * self.r * self.r


def total_area(items: Iterable[Any]) -> float:
    """
    EAFP style: trust objects to have area(); handle AttributeError if not.
    """
    acc = 0.0
    for obj in items:
        try:
            acc += float(obj.area())  # type: ignore[attr-defined]
        except AttributeError as e:
            raise TypeError(f"Object {obj!r} does not support area()") from e
    return acc


def notify_start(engine: Any) -> str:
    """
    Another duck-typed API: object must implement .start() -> str
    """
    if not hasattr(engine, "start"):
        raise TypeError("Object must implement start()")
    return engine.start()  # type: ignore[no-any-return]


class DieselEngine:
    def start(self) -> str:
        return "DieselEngine: rumble..."


class ElectricMotor:
    def start(self) -> str:
        return "ElectricMotor: whirr..."


def demo_duck_typing(out: Out) -> None:
    out.rule("Duck Typing (behavior over types)")
    out.h2("Example 1: total_area() with various 'shape-like' objects")
    shapes = [Circle(1.5), Rectangle(2, 3), ThirdPartyDisk(1.0)]
    out.info(f"total_area = {total_area(shapes):.3f}")

    out.h2("Example 2: notify_start() expects .start() regardless of class lineage")
    out.info(notify_start(DieselEngine()))
    out.info(notify_start(ElectricMotor()))
    try:
        notify_start(object())  # no start()
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")
    out.ok("Duck typing focuses on available behavior, not inheritance.")


# ==============================================================================
# 8) Magic / Dunder Methods
# ==============================================================================

class Polynomial:
    """
    Polynomial with multiple dunder methods:
    - __str__/__repr__   : readable representation
    - __eq__             : value equality
    - __add__/__mul__    : arithmetic on polynomials
    - __call__           : evaluate at x
    - __len__            : number of coefficients

    Example 1:
        INPUT:
            p = Polynomial([1, 0, 2])   # 1 + 0x + 2x^2
            q = Polynomial([0, 3])      # 3x
            str(p), len(p), p(2)
            (p + q), (p * q)
        EXPECTED:
            "2x^2 + 1", 3, 9
            sum and product with correct coefficients

    Example 2 (__eq__ and normalization):
        INPUT:
            Polynomial([0, 1, 0]) == Polynomial([0, 1])
        EXPECTED:
            True (trailing zeros normalized away)
    """
    def __init__(self, coeffs: Sequence[Union[int, float]]) -> None:
        if not isinstance(coeffs, (list, tuple)) or not coeffs:
            raise ValueError("coeffs must be a non-empty sequence.")
        self._coeffs: List[float] = [float(c) for c in coeffs]
        self._normalize()

    def _normalize(self) -> None:
        # remove trailing zeros to canonical form; keep at least one
        while len(self._coeffs) > 1 and abs(self._coeffs[-1]) < 1e-12:
            self._coeffs.pop()

    def __len__(self) -> int:
        return len(self._coeffs)

    def __call__(self, x: float) -> float:
        # Horner's method for evaluation
        acc = 0.0
        for c in reversed(self._coeffs):
            acc = acc * x + c
        return acc

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self._coeffs == other._coeffs

    def __add__(self, other: Any) -> "Polynomial":
        if not isinstance(other, Polynomial):
            return NotImplemented
        n = max(len(self), len(other))
        a = self._coeffs + [0.0] * (n - len(self))
        b = other._coeffs + [0.0] * (n - len(other))
        return Polynomial([x + y for x, y in zip(a, b)])

    def __mul__(self, other: Any) -> "Polynomial":
        if not isinstance(other, Polynomial):
            return NotImplemented
        res = [0.0] * (len(self) + len(other) - 1)
        for i, a in enumerate(self._coeffs):
            for j, b in enumerate(other._coeffs):
                res[i + j] += a * b
        return Polynomial(res)

    def __repr__(self) -> str:
        return f"Polynomial({self._coeffs!r})"

    def __str__(self) -> str:
        terms: List[str] = []
        for i, c in reversed(list(enumerate(self._coeffs))):
            if abs(c) < 1e-12:
                continue
            if i == 0:
                terms.append(f"{c:.0f}".rstrip("0").rstrip("."))
            elif i == 1:
                terms.append(f"{c:.0f}x".rstrip("0").rstrip(".").replace("1x", "x"))
            else:
                s = f"{c:.0f}x^{i}".rstrip("0").rstrip(".").replace("1x", "x")
                terms.append(s)
        if not terms:
            return "0"
        return " + ".join(terms).replace("+ -", "- ")


class Bag:
    """
    Multiset using dunder methods:
    - __len__, __contains__, __iter__
    - __add__ (multiset union by count), __sub__ (difference)
    - __str__/__repr__

    Example 1:
        INPUT:
            b1 = Bag("banana"); b2 = Bag("bandana")
            len(b1), 'a' in b1
            (b1 + b2), (b2 - b1)
        EXPECTED:
            counts combined/subtracted correctly

    Example 2 (iteration):
        INPUT:
            list(iter(Bag("aba")))
        EXPECTED:
            elements expanded according to counts, e.g., ['a','a','b']
    """
    def __init__(self, iterable: Iterable[Any] = ()) -> None:
        self._count: Dict[Any, int] = {}
        for x in iterable:
            self._count[x] = self._count.get(x, 0) + 1

    def __len__(self) -> int:
        return sum(self._count.values())

    def __contains__(self, item: Any) -> bool:
        return self._count.get(item, 0) > 0

    def __iter__(self):
        for item, n in self._count.items():
            for _ in range(n):
                yield item

    def __add__(self, other: Any) -> "Bag":
        if not isinstance(other, Bag):
            return NotImplemented
        res = Bag()
        for k, v in self._count.items():
            res._count[k] = res._count.get(k, 0) + v
        for k, v in other._count.items():
            res._count[k] = res._count.get(k, 0) + v
        return res

    def __sub__(self, other: Any) -> "Bag":
        if not isinstance(other, Bag):
            return NotImplemented
        res = Bag()
        for k, v in self._count.items():
            res._count[k] = max(0, v - other._count.get(k, 0))
        return res

    def __repr__(self) -> str:
        return f"Bag({self._count!r})"

    def __str__(self) -> str:
        items = ", ".join(f"{k}:{v}" for k, v in sorted(self._count.items(), key=lambda kv: kv[0]))
        return f"Bag[{items}]"


def demo_dunder_methods(out: Out) -> None:
    out.rule("Magic / Dunder Methods")
    out.h2("Example 1: Polynomial arithmetic, evaluation, equality, stringification")
    p = Polynomial([1, 0, 2])  # 1 + 0x + 2x^2
    q = Polynomial([0, 3])     # 3x
    out.info(f"str(p)={str(p)}, len(p)={len(p)}, p(2)={p(2)}")
    out.info(f"p+q -> {p + q}")
    out.info(f"p*q -> {p * q}")
    out.info(f"Polynomial([0, 1, 0]) == Polynomial([0, 1]) -> {Polynomial([0, 1, 0]) == Polynomial([0, 1])}")

    out.h2("Example 2: Bag multiset dunders")
    b1 = Bag("banana")
    b2 = Bag("bandana")
    out.info(f"len(b1)={len(b1)}, 'a' in b1 -> {'a' in b1}")
    out.info(f"b1 + b2 -> {b1 + b2}")
    out.info(f"b2 - b1 -> {b2 - b1}")
    out.info(f"iter(Bag('aba')) -> {list(Bag('aba'))}")
    out.ok("Dunder methods tailor behavior to Python operators/protocols.")


# ==============================================================================
# 9) Abstract Base Classes (ABC)
# ==============================================================================

class Plugin(ABC):
    """
    ABC defines interface for plugins: name and run(data) -> str.

    Example 1:
        INPUT:
            p = EchoPlugin()
            p.run('hello')
        EXPECTED:
            'hello'

        INPUT (cannot instantiate ABC):
            Plugin()
        EXPECTED:
            TypeError

    Example 2:
        INPUT:
            p = UpperPlugin()
            p.run('hi')
        EXPECTED:
            'HI'
    """
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, data: str) -> str:
        ...


class EchoPlugin(Plugin):
    @property
    def name(self) -> str:
        return "echo"

    def run(self, data: str) -> str:
        return data


class UpperPlugin(Plugin):
    @property
    def name(self) -> str:
        return "upper"

    def run(self, data: str) -> str:
        return data.upper()


class DataSource(ABC):
    """
    Another ABC example with read/write abstraction.
    """
    @abstractmethod
    def read(self) -> str:
        ...

    @abstractmethod
    def write(self, data: str) -> None:
        ...


class MemoryDataSource(DataSource):
    def __init__(self) -> None:
        self._buf = ""

    def read(self) -> str:
        return self._buf

    def write(self, data: str) -> None:
        self._buf += data


def demo_abc(out: Out) -> None:
    out.rule("Abstract Base Classes (ABC)")
    out.h2("Example 1: Plugin ABC with concrete implementations")
    echo = EchoPlugin()
    upper = UpperPlugin()
    out.info(f"{echo.name} -> {echo.run('hello')}")
    out.info(f"{upper.name} -> {upper.run('hi')}")

    try:
        _ = Plugin()  # type: ignore[abstract]
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    out.h2("Example 2: DataSource abstraction")
    src = MemoryDataSource()
    src.write("abc")
    out.info(f"MemoryDataSource.read() -> {src.read()}")
    out.ok("ABCs enforce interface; subclasses provide implementations.")


# ==============================================================================
# 10) Metaclasses — classes that create classes
# ==============================================================================

class ValidatedMeta(type):
    """
    Metaclass to:
    - Enforce that each class has a docstring.
    - Forbid attributes starting with 'bad_' (demo rule).
    - Auto-inject __repr__ if absent (lists public attributes).
    - Keep a registry of created classes.

    Example 1 (good class):
        INPUT:
            class Config(metaclass=ValidatedMeta):
                "Holds configuration."
                def __init__(self, host: str, port: int) -> None:
                    self.host = host; self.port = port
            c = Config("localhost", 8080); repr(c)
        EXPECTED:
            "Config(host='localhost', port=8080)"

    Example 2 (bad classes):
        INPUT:
            class Bad1(metaclass=ValidatedMeta): pass
        EXPECTED:
            TypeError (missing docstring)

        INPUT:
            class Bad2(metaclass=ValidatedMeta):
                "Doc"
                bad_attr = 1
        EXPECTED:
            TypeError (forbidden attribute)
    """
    registry: Dict[str, type] = {}

    def __new__(mcls, name: str, bases: Tuple[type, ...], ns: Dict[str, Any], **kw: Any):
        # Enforce docstring presence
        doc = ns.get("__doc__", None)
        if not doc or not doc.strip():
            raise TypeError(f"{name} must have a non-empty docstring.")

        # Forbid 'bad_' attributes
        for attr in ns:
            if attr.startswith("bad_"):
                raise TypeError(f"{name} defines forbidden attribute '{attr}'")

        cls = super().__new__(mcls, name, bases, ns, **kw)

        # Inject __repr__ if absent
        if "__repr__" not in ns:
            def __repr__(self) -> str:  # type: ignore[no-redef]
                # show only public attributes
                attrs = [k for k in dir(self) if not k.startswith("_") and not callable(getattr(self, k, None))]
                kv = ", ".join(f"{k}={getattr(self, k)!r}" for k in attrs)
                return f"{type(self).__name__}({kv})"
            setattr(cls, "__repr__", __repr__)

        # Register class
        ValidatedMeta.registry[name] = cls
        return cls


def demo_metaclass(out: Out) -> None:
    out.rule("Metaclasses (custom class creation)")
    out.h2("Example 1: Valid class auto-gets a helpful __repr__")

    class Config(metaclass=ValidatedMeta):
        "Holds configuration."
        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port

    c = Config("localhost", 8080)
    out.info(f"repr(c) -> {repr(c)}")
    out.info(f"Registry contains: {list(ValidatedMeta.registry)}")

    out.h2("Example 2: Violations caught at class creation time")
    try:
        class Bad1(metaclass=ValidatedMeta):
            pass
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    try:
        class Bad2(metaclass=ValidatedMeta):
            "Doc present"
            bad_attr = 1
    except Exception as e:
        out.err(f"{type(e).__name__}: {e}")

    out.ok("Metaclass enforced rules and injected behavior at class creation.")


# ==============================================================================
# CLI Harness
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced OOP features in Python — single-file, professional tutorial.")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity.")
    parser.add_argument("--no-rich", action="store_true", help="Force plain output even if 'rich' is installed.")
    parser.add_argument(
        "--only",
        choices=[
            "constructor",
            "destructor",
            "self",
            "classmethod",
            "staticmethod",
            "properties",
            "duck",
            "dunder",
            "abc",
            "metaclass",
            "all",
        ],
        default="all",
        help="Run a specific topic only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Out(verbose=not args.quiet, use_rich=(not args.no_rich))

    out.rule("⚙️ Advanced OOP Features in Python — Mastery Edition")

    if args.only in ("constructor", "all"):
        demo_constructor(out)
        out.print()

    if args.only in ("destructor", "all"):
        demo_destructor(out)
        out.print()

    if args.only in ("self", "all"):
        demo_self(out)
        out.print()

    if args.only in ("classmethod", "all"):
        demo_classmethod(out)
        out.print()

    if args.only in ("staticmethod", "all"):
        demo_staticmethod(out)
        out.print()

    if args.only in ("properties", "all"):
        demo_properties(out)
        out.print()

    if args.only in ("duck", "all"):
        demo_duck_typing(out)
        out.print()

    if args.only in ("dunder", "all"):
        demo_dunder_methods(out)
        out.print()

    if args.only in ("abc", "all"):
        demo_abc(out)
        out.print()

    if args.only in ("metaclass", "all"):
        demo_metaclass(out)
        out.print()

    out.rule("DONE")
    out.ok("All requested sections executed successfully.")


if __name__ == "__main__":
    main()