"""
advanced_oop_masterclass.py — Single-file, deeply commented Python tutorial on:
- Constructor (__init__)
- Destructor (__del__)
- self keyword
- Class Method (@classmethod)
- Static Method (@staticmethod)
- Properties & Getters/Setters (@property)
- Duck Typing
- Magic / Dunder Methods (__str__, __len__, __eq__, __add__, etc.)
- Abstract Base Classes (ABC)
- Metaclasses

Design notes:
- All explanations are embedded as comments/docstrings next to the code they explain.
- Each demo function includes "Input" and "Expected output" comments.
- Uses 'rich' for professional output if installed; otherwise falls back to print().
- Toggle VERBOSE to control console verbosity.

Python: 3.9+ (tested with 3.11)
"""

from __future__ import annotations

import gc
import io
import json
import math
import re
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple


# ------------------------------------------------------------------------------
# Configuration: verbosity and optional rich console setup
# ------------------------------------------------------------------------------
VERBOSE: bool = True  # Set False to silence demo prints.

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _console = Console()

    def _print_header(title: str) -> None:
        if VERBOSE:
            _console.rule(Text(title, style="bold cyan"))

    def _print(msg: Any) -> None:
        if VERBOSE:
            _console.print(msg)

    def _print_panel(title: str, body: str) -> None:
        if VERBOSE:
            _console.print(Panel(body, title=title, border_style="cyan", expand=False))

except Exception:
    _console = None

    def _print_header(title: str) -> None:
        if VERBOSE:
            print("\n" + "=" * 10 + f" {title} " + "=" * 10)

    def _print(msg: Any) -> None:
        if VERBOSE:
            print(msg)

    def _print_panel(title: str, body: str) -> None:
        if VERBOSE:
            print("\n" + "=" * 10 + f" {title} " + "=" * 10)
            print(body)


# ==============================================================================
# Section 1 — Constructor (__init__)
# ==============================================================================
class User:
    """
    Example 1: Constructor that validates inputs and sets up state.

    - __init__ is called automatically when you instantiate the class.
    - We validate 'username' and 'email'; an invalid input raises ValueError.
    - Demonstrates a lightweight, explicit constructor.

    Exceptions:
    - ValueError on empty username or invalid email format.
    """

    EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

    def __init__(self, username: str, email: str) -> None:
        if not username or not username.strip():
            raise ValueError("username must be a non-empty string.")
        if not email or not self.EMAIL_RE.match(email):
            raise ValueError("email is invalid.")
        self._username: str = username.strip()
        self._email: str = email.strip()
        self._joined: date = date.today()

    def __repr__(self) -> str:
        return f"User(username='{self._username}', email='{self._email}', joined={self._joined.isoformat()})"


class Matrix:
    """
    Example 2: Constructor building a 2D matrix from nested sequences.

    - Validates rectangular shape and numeric entries.
    - Stores shape and provides a transpose() method to show that the object is ready to use.

    Exceptions:
    - ValueError for empty input or ragged rows.
    - TypeError if any element is not int/float.
    """

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        if not rows:
            raise ValueError("Matrix cannot be empty.")
        row_lengths = {len(r) for r in rows}
        if len(row_lengths) != 1:
            raise ValueError("All rows must have the same length (rectangular matrix).")
        # Validate elements:
        for r in rows:
            for x in r:
                if not isinstance(x, (int, float)):
                    raise TypeError("Matrix elements must be numeric.")
        self._m: List[List[float]] = [list(map(float, r)) for r in rows]
        self.nrows: int = len(self._m)
        self.ncols: int = len(self._m[0])

    def transpose(self) -> "Matrix":
        return Matrix(list(zip(*self._m)))

    def __repr__(self) -> str:
        return f"Matrix({self._m})"


def demo_constructors() -> None:
    """
    Input:
        u = User("alice", "alice@example.com")
        print(u)

        m = Matrix([[1, 2, 3], [4, 5, 6]])
        print(m)
        print(m.transpose())

        try: User("", "bad")        # ValueError
        except ValueError: print("invalid")

        try: Matrix([[1, 2], [3]])  # ValueError ragged
        except ValueError: print("ragged")

    Expected output:
        User(username='alice', email='alice@example.com', joined=YYYY-MM-DD)
        Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Matrix([(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)])
        invalid
        ragged
    """
    _print_header("Constructors — __init__")

    u = User("alice", "alice@example.com")
    _print(u)

    m = Matrix([[1, 2, 3], [4, 5, 6]])
    _print(m)
    _print(m.transpose())

    try:
        User("", "bad")
    except ValueError:
        _print("invalid")

    try:
        Matrix([[1, 2], [3]])
    except ValueError:
        _print("ragged")

    assert "alice" in repr(u)
    assert m.nrows == 2 and m.ncols == 3


# ==============================================================================
# Section 2 — Destructor (__del__)
# ==============================================================================
_DELETIONS: int = 0  # global counter to track __del__ runs (demo only)


class Tracker:
    """
    Example 1: A tiny object with a destructor that bumps a global counter.

    WARNING: __del__ timing is non-deterministic in general (implementation-dependent).
    In CPython, objects may be destroyed immediately when the refcount hits zero;
    other interpreters might delay it until garbage collection.

    We force a GC collection in the demo to increase the chance of seeing __del__.
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __del__(self) -> None:
        global _DELETIONS
        _DELETIONS += 1  # side-effect visible to the demo


class SafeHandle:
    """
    Example 2: __del__ as a last resort; the preferred pattern is explicit close() or a context manager.

    - Simulate a resource that needs closing.
    - __del__ closes if the user forgot; we also implement __enter__/__exit__ to prefer with-statements.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._open = True

    def __enter__(self) -> "SafeHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._open:
            self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def __del__(self) -> None:
        # Not guaranteed when or if this runs, but helpful as a safety net.
        try:
            self.close()
        except Exception:
            # Avoid raising from __del__ (it would be ignored and can cause noisy logs).
            pass


def demo_destructors_tracker() -> None:
    """
    Input:
        global _DELETIONS
        _DELETIONS = 0
        for i in range(5):
            Tracker(f"T{i}")
        gc.collect()
        print(_DELETIONS >= 1)

    Expected output:
        True   # (At least some destructions observed after GC)
    """
    _print_header("Destructors — __del__ (Tracker)")

    global _DELETIONS
    _DELETIONS = 0

    for i in range(5):
        Tracker(f"T{i}")  # no reference kept

    gc.collect()  # Encourage finalization (demo)
    _print(f"Destructors observed: {_DELETIONS} (>=1 expected)")

    assert _DELETIONS >= 1


def demo_destructors_safehandle() -> None:
    """
    Input:
        with SafeHandle("session") as h:
            print(h.is_open)  # True
        print(h.is_open)      # False (closed by __exit__)

        handle = SafeHandle("leaky")
        print(handle.is_open) # True
        del handle
        gc.collect()
        # handle name no longer exists; __del__ should have run, closing it

    Expected output:
        True
        False
        (no NameError printed here – just demonstration)
    """
    _print_header("Destructors — __del__ (SafeHandle)")

    with SafeHandle("session") as h:
        _print(h.is_open)
    _print(h.is_open)

    handle = SafeHandle("leaky")
    _print(handle.is_open)
    del handle
    gc.collect()  # Encourage finalization


# ==============================================================================
# Section 3 — The self keyword
# ==============================================================================
class FluentCounter:
    """
    Example 1: 'self' is the current instance; return self for fluent APIs.
    """

    def __init__(self, start: int = 0) -> None:
        self._value = int(start)

    def inc(self, step: int = 1) -> "FluentCounter":
        self._value += step
        return self  # allows c.inc().inc(2).dec()

    def dec(self, step: int = 1) -> "FluentCounter":
        self._value -= step
        return self

    @property
    def value(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"FluentCounter(value={self._value})"


class Shadowing:
    """
    Example 2: 'self' resolves instance attributes even when a class attribute shares the name.
    """

    name: str = "CLASS"  # class attribute

    def __init__(self, name: str) -> None:
        self.name = name  # instance attribute shadows class attribute

    def describe(self) -> str:
        return f"instance={self.name}, class={Shadowing.name}"


def demo_self_keyword_1() -> None:
    """
    Input:
        c = FluentCounter().inc().inc(2).dec()
        print(c, c.value)

    Expected output:
        FluentCounter(value=2) 2
    """
    _print_header("self — fluent API")

    c = FluentCounter().inc().inc(2).dec()
    _print(f"{c} {c.value}")
    assert c.value == 2


def demo_self_keyword_2() -> None:
    """
    Input:
        s = Shadowing("INSTANCE")
        print(s.describe())

    Expected output:
        instance=INSTANCE, class=CLASS
    """
    _print_header("self — attribute resolution")

    s = Shadowing("INSTANCE")
    _print(s.describe())
    assert s.describe() == "instance=INSTANCE, class=CLASS"


# ==============================================================================
# Section 4 — Class Method (@classmethod)
# ==============================================================================
class SimpleDate:
    """
    Example 1: Alternate constructors via classmethod.
    """

    def __init__(self, year: int, month: int, day: int) -> None:
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)

    @classmethod
    def from_iso(cls, s: str) -> "SimpleDate":
        y, m, d = map(int, s.split("-"))
        return cls(y, m, d)

    @classmethod
    def today(cls) -> "SimpleDate":
        t = date.today()
        return cls(t.year, t.month, t.day)

    def __repr__(self) -> str:
        return f"SimpleDate({self.year:04d}-{self.month:02d}-{self.day:02d})"


class Session:
    """
    Example 2: Class-wide counters managed with classmethod.

    - Tracks active session count across all instances.
    """

    _active_count: int = 0

    def __init__(self) -> None:
        type(self)._active_count += 1

    def close(self) -> None:
        if self.active_count() > 0:
            type(self)._active_count -= 1

    @classmethod
    def active_count(cls) -> int:
        return cls._active_count


def demo_classmethod_1() -> None:
    """
    Input:
        d1 = SimpleDate.from_iso("2025-09-11")
        d2 = SimpleDate.today()
        print(d1, isinstance(d1, SimpleDate))
        print(d2)

    Expected output:
        SimpleDate(2025-09-11) True
        SimpleDate(YYYY-MM-DD)
    """
    _print_header("classmethod — alternate constructors")

    d1 = SimpleDate.from_iso("2025-09-11")
    d2 = SimpleDate.today()
    _print(f"{d1} {isinstance(d1, SimpleDate)}")
    _print(str(d2))
    assert isinstance(d1, SimpleDate)


def demo_classmethod_2() -> None:
    """
    Input:
        s1, s2 = Session(), Session()
        print(Session.active_count())
        s1.close()
        print(Session.active_count())

    Expected output:
        2
        1
    """
    _print_header("classmethod — class-wide counters")

    s1, s2 = Session(), Session()
    _print(Session.active_count())
    s1.close()
    _print(Session.active_count())
    assert Session.active_count() == 1
    s2.close()


# ==============================================================================
# Section 5 — Static Method (@staticmethod)
# ==============================================================================
class Validators:
    """
    Example 1: Static methods as utility functions; no access to cls/self.
    """

    EMAIL_RE = User.EMAIL_RE

    @staticmethod
    def is_email(s: str) -> bool:
        return bool(Validators.EMAIL_RE.match(s))

    @staticmethod
    def is_phone(s: str) -> bool:
        # A toy phone check: digits with optional leading + and 7..15 digits
        return bool(re.fullmatch(r"\+?\d{7,15}", s))


class MathTools:
    """
    Example 2: Static numerical helpers.
    """

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        if lo > hi:
            raise ValueError("lo cannot be greater than hi.")
        return min(max(x, lo), hi)

    @staticmethod
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        r = int(math.sqrt(n))
        f = 3
        while f <= r:
            if n % f == 0:
                return False
            f += 2
        return True


def demo_staticmethod_1() -> None:
    """
    Input:
        print(Validators.is_email("a@b.com"))
        print(Validators.is_phone("+1234567890"))

    Expected output:
        True
        True
    """
    _print_header("staticmethod — validators")

    _print(Validators.is_email("a@b.com"))
    _print(Validators.is_phone("+1234567890"))

    assert Validators.is_email("a@b.com")
    assert not Validators.is_email("oops")


def demo_staticmethod_2() -> None:
    """
    Input:
        print(MathTools.clamp(10, 0, 5))
        print(MathTools.is_prime(29), MathTools.is_prime(1))

    Expected output:
        5
        True False
    """
    _print_header("staticmethod — math tools")

    _print(MathTools.clamp(10, 0, 5))
    _print(MathTools.is_prime(29), MathTools.is_prime(1))

    assert MathTools.clamp(-1, 0, 5) == 0
    assert MathTools.is_prime(29) and not MathTools.is_prime(1)


# ==============================================================================
# Section 6 — Properties & Getters/Setters (@property)
# ==============================================================================
class Temperature:
    """
    Example 1: Properties with validation and computed views.
    - celsius (storage)
    - fahrenheit (computed; backed by celsius)
    - kelvin (read-only computed)
    """

    def __init__(self, celsius: float) -> None:
        self.celsius = celsius  # uses setter for validation

    @property
    def celsius(self) -> float:
        return self._c

    @celsius.setter
    def celsius(self, value: float) -> None:
        v = float(value)
        if v < -273.15:
            raise ValueError("Temperature below absolute zero.")
        self._c = v

    @property
    def fahrenheit(self) -> float:
        return self._c * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, f: float) -> None:
        self.celsius = (float(f) - 32) * 5 / 9

    @property
    def kelvin(self) -> float:
        return self._c + 273.15


class Account:
    """
    Example 2: Property-based encapsulation with read-only and write-validated fields.

    - balance is read-only; deposit/withdraw control updates.
    - owner has a setter with validation.
    - nickname supports a deleter to clear it.
    """

    def __init__(self, owner: str, opening_balance: float = 0.0) -> None:
        if not owner.strip():
            raise ValueError("owner cannot be empty.")
        if opening_balance < 0:
            raise ValueError("opening_balance cannot be negative.")
        self._owner = owner.strip()
        self._balance = float(opening_balance)
        self._nickname: Optional[str] = None

    @property
    def owner(self) -> str:
        return self._owner

    @owner.setter
    def owner(self, value: str) -> None:
        if not value.strip():
            raise ValueError("owner cannot be empty.")
        self._owner = value.strip()

    @property
    def balance(self) -> float:
        return self._balance

    def deposit(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("amount must be positive.")
        self._balance += amount
        return self._balance

    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("amount must be positive.")
        if amount > self._balance:
            raise ValueError("insufficient funds.")
        self._balance -= amount
        return self._balance

    @property
    def nickname(self) -> Optional[str]:
        return self._nickname

    @nickname.setter
    def nickname(self, value: Optional[str]) -> None:
        if value is not None and not value.strip():
            raise ValueError("nickname cannot be empty if provided.")
        self._nickname = None if value is None else value.strip()

    @nickname.deleter
    def nickname(self) -> None:  # type: ignore[override]
        self._nickname = None


def demo_properties_1() -> None:
    """
    Input:
        t = Temperature(0)
        print(round(t.fahrenheit, 1))
        t.fahrenheit = 212
        print(round(t.celsius, 1), round(t.kelvin, 2))
        try: t.celsius = -300
        except ValueError: print("invalid")

    Expected output:
        32.0
        100.0 373.15
        invalid
    """
    _print_header("properties — temperature conversions")

    t = Temperature(0)
    _print(round(t.fahrenheit, 1))
    t.fahrenheit = 212
    _print(round(t.celsius, 1), round(t.kelvin, 2))

    try:
        t.celsius = -300
    except ValueError:
        _print("invalid")

    assert math.isclose(t.celsius, 100.0)


def demo_properties_2() -> None:
    """
    Input:
        a = Account("Alice", 100)
        a.deposit(50)
        print(a.balance)
        a.nickname = "Al"
        print(a.nickname)
        del a.nickname
        print(a.nickname)

    Expected output:
        150.0
        Al
        None
    """
    _print_header("properties — account encapsulation")

    a = Account("Alice", 100)
    a.deposit(50)
    _print(a.balance)
    a.nickname = "Al"
    _print(a.nickname)
    del a.nickname
    _print(a.nickname)
    assert a.balance == 150.0 and a.nickname is None


# ==============================================================================
# Section 7 — Duck Typing
# ==============================================================================
def make_it_quack(obj: Any) -> str:
    """
    Example 1 helper: EAFP (Easier to Ask Forgiveness than Permission).
    Calls obj.quack() if available; raises AttributeError otherwise.
    """
    quack = getattr(obj, "quack")  # might raise AttributeError if missing
    return quack()


class Duck:
    def quack(self) -> str:
        return "quack"


class RobotDuck:
    def quack(self) -> str:
        return "beep-quack"


class NoQuack:
    pass


def demo_duck_typing_1() -> None:
    """
    Input:
        print(make_it_quack(Duck()))
        print(make_it_quack(RobotDuck()))
        try: print(make_it_quack(NoQuack()))
        except AttributeError: print("no-quack")

    Expected output:
        quack
        beep-quack
        no-quack
    """
    _print_header("duck typing — behavior over type")

    _print(make_it_quack(Duck()))
    _print(make_it_quack(RobotDuck()))
    try:
        _print(make_it_quack(NoQuack()))
    except AttributeError:
        _print("no-quack")


class ListWriter:
    """
    Example 2: Any 'writer' with a .write(str) method works.
    Here, we capture written strings in a list.
    """

    def __init__(self) -> None:
        self.data: List[str] = []

    def write(self, s: str) -> int:
        self.data.append(s)
        return len(s)


def write_report(sink: Any, text: str) -> int:
    """
    Duck-typed writer: accepts anything with a .write(str) -> int method.
    """
    return sink.write(text)


def demo_duck_typing_2() -> None:
    """
    Input:
        mem = io.StringIO()
        lw = ListWriter()
        write_report(mem, "hello")
        write_report(lw, "world")
        print(mem.getvalue())
        print(lw.data)

    Expected output:
        hello
        ['world']
    """
    _print_header("duck typing — file-like writers")

    mem = io.StringIO()
    lw = ListWriter()
    write_report(mem, "hello")
    write_report(lw, "world")
    _print(mem.getvalue())
    _print(lw.data)
    assert mem.getvalue() == "hello" and lw.data == ["world"]


# ==============================================================================
# Section 8 — Magic / Dunder Methods
# ==============================================================================
class Polynomial:
    """
    Example 1: Demonstrates __str__, __repr__, __len__, __eq__, __add__, __call__.

    Represent a polynomial a_n x^n + ... + a_1 x + a_0 using coefficients list [a_n,...,a_0].
    """

    def __init__(self, coeffs: Sequence[float]) -> None:
        if not coeffs:
            raise ValueError("coeffs cannot be empty.")
        self.coeffs: Tuple[float, ...] = tuple(float(c) for c in coeffs)
        # Normalize leading zeros
        i = 0
        while i < len(self.coeffs) - 1 and math.isclose(self.coeffs[i], 0.0):
            i += 1
        self.coeffs = self.coeffs[i:]

    def __len__(self) -> int:
        return len(self.coeffs)  # number of terms (not degree+1 if leading zeros removed)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return all(math.isclose(a, b) for a, b in zip(self.coeffs, other.coeffs)) and len(self) == len(other)

    def __add__(self, other: Any) -> "Polynomial":
        if not isinstance(other, Polynomial):
            return NotImplemented
        a, b = list(self.coeffs), list(other.coeffs)
        # Pad shorter on the left
        if len(a) < len(b):
            a = [0.0] * (len(b) - len(a)) + a
        elif len(b) < len(a):
            b = [0.0] * (len(a) - len(b)) + b
        return Polynomial([x + y for x, y in zip(a, b)])

    def __call__(self, x: float) -> float:
        # Horner's method
        acc = 0.0
        for c in self.coeffs:
            acc = acc * x + c
        return acc

    def __repr__(self) -> str:
        return f"Polynomial({list(self.coeffs)!r})"

    def __str__(self) -> str:
        # Human-friendly form
        terms: List[str] = []
        n = len(self.coeffs) - 1
        for i, c in enumerate(self.coeffs):
            power = n - i
            if math.isclose(c, 0.0):
                continue
            if power == 0:
                terms.append(f"{c:.3g}")
            elif power == 1:
                terms.append(f"{c:.3g}x")
            else:
                terms.append(f"{c:.3g}x^{power}")
        return " + ".join(terms) if terms else "0"


class Bag:
    """
    Example 2: Multiset with __len__, __contains__, __add__, __eq__, __iter__, __str__.
    """

    def __init__(self, items: Optional[Iterable[Any]] = None) -> None:
        self._counts: Dict[Any, int] = {}
        if items is not None:
            for it in items:
                self._counts[it] = self._counts.get(it, 0) + 1

    def __len__(self) -> int:
        return sum(self._counts.values())

    def __contains__(self, item: Any) -> bool:
        return self._counts.get(item, 0) > 0

    def __add__(self, other: Any) -> "Bag":
        if not isinstance(other, Bag):
            return NotImplemented
        result = Bag()
        # Merge counts
        keys = set(self._counts) | set(other._counts)
        for k in keys:
            c = self._counts.get(k, 0) + other._counts.get(k, 0)
            if c:
                result._counts[k] = c
        return result

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Bag):
            return NotImplemented
        return self._counts == other._counts

    def __iter__(self) -> Iterator[Any]:
        for k, c in self._counts.items():
            for _ in range(c):
                yield k

    def __repr__(self) -> str:
        return f"Bag({self._counts!r})"

    def __str__(self) -> str:
        items = ", ".join(f"{k}×{c}" for k, c in sorted(self._counts.items(), key=lambda kv: str(kv[0])))
        return "{" + items + "}"


def demo_dunder_1() -> None:
    """
    Input:
        p1 = Polynomial([1, 0, -2])   # x^2 - 2
        p2 = Polynomial([0, 3])       # 3
        p3 = p1 + p2                  # x^2 + 1
        print(str(p3))
        print(round(p3(2), 2))
        print(len(p3))

    Expected output:
        1x^2 + 1
        5.0
        2
    """
    _print_header("dunder — Polynomial")

    p1 = Polynomial([1, 0, -2])  # x^2 - 2
    p2 = Polynomial([0, 3])      # 3
    p3 = p1 + p2                  # x^2 + 1
    _print(str(p3))
    _print(round(p3(2), 2))
    _print(len(p3))

    assert str(p3).startswith("1x^2")
    assert math.isclose(p3(2), 5.0)


def demo_dunder_2() -> None:
    """
    Input:
        b1 = Bag(["a", "b", "a"])
        b2 = Bag(["b", "c"])
        b3 = b1 + b2
        print(str(b3))
        print(len(b3), "a" in b3, list(sorted(b3)))

    Expected output:
        {a×2, b×2, c×1}
        5 True ['a', 'a', 'b', 'b', 'c']
    """
    _print_header("dunder — Bag (multiset)")

    b1 = Bag(["a", "b", "a"])
    b2 = Bag(["b", "c"])
    b3 = b1 + b2
    _print(str(b3))
    _print(len(b3), "a" in b3, list(sorted(b3)))
    assert len(b3) == 5 and "a" in b3


# ==============================================================================
# Section 9 — Abstract Base Classes (ABC)
# ==============================================================================
class Report(ABC):
    """
    Example 1: Abstract Base Class with abstract method and abstract property.

    Contract:
    - format (property): short format name
    - render(data) -> str: produce a string representation
    """

    @property
    @abstractmethod
    def format(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def render(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError

    def header(self, title: str) -> str:
        # Optional reusable piece of behavior
        return f"[Report: {title}]"


class TextReport(Report):
    @property
    def format(self) -> str:
        return "text"

    def render(self, data: Dict[str, Any]) -> str:
        lines = [self.header("TEXT")]
        for k, v in data.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)


class HtmlReport(Report):
    @property
    def format(self) -> str:
        return "html"

    def render(self, data: Dict[str, Any]) -> str:
        body = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in data.items())
        return f"<h1>{self.header('HTML')}</h1><ul>{body}</ul>"


def demo_abc_1() -> None:
    """
    Input:
        try:
            Report()  # TypeError: can't instantiate abstract class
        except TypeError: print("cannot instantiate ABC")

        r1 = TextReport()
        r2 = HtmlReport()
        print(r1.format, r2.format)
        print(r1.render({"a": 1}))
        print(r2.render({"b": 2}))

    Expected output:
        cannot instantiate ABC
        text html
        [Report: TEXT]
        a: 1
        <h1>[Report: HTML]</h1><ul><li><b>b</b>: 2</li></ul>
    """
    _print_header("ABC — Reports")

    try:
        Report()  # type: ignore[abstract]
    except TypeError:
        _print("cannot instantiate ABC")

    r1, r2 = TextReport(), HtmlReport()
    _print(f"{r1.format} {r2.format}")
    _print(r1.render({"a": 1}))
    _print(r2.render({"b": 2}))
    assert r1.format == "text" and r2.format == "html"


class DataSource(ABC):
    """
    Example 2: ABC for a simple data source with read/write/closed.
    """

    @abstractmethod
    def read(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def write(self, s: str) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def closed(self) -> bool:
        raise NotImplementedError


class MemorySource(DataSource):
    def __init__(self) -> None:
        self._buf = io.StringIO()
        self._closed = False

    def read(self) -> str:
        if self._closed:
            raise RuntimeError("source is closed")
        return self._buf.getvalue()

    def write(self, s: str) -> int:
        if self._closed:
            raise RuntimeError("source is closed")
        return self._buf.write(s)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True


class ReadOnlySource(DataSource):
    def __init__(self, data: str) -> None:
        self._data = data
        self._closed = False

    def read(self) -> str:
        if self._closed:
            raise RuntimeError("source is closed")
        return self._data

    def write(self, s: str) -> int:
        raise PermissionError("read-only")

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True


def demo_abc_2() -> None:
    """
    Input:
        ms = MemorySource()
        ms.write("hi")
        print(ms.read())
        ms.close()
        try: ms.read()
        except RuntimeError: print("closed")

        rs = ReadOnlySource("snap")
        print(rs.read())
        try: rs.write("x")
        except PermissionError: print("ro")

    Expected output:
        hi
        closed
        snap
        ro
    """
    _print_header("ABC — DataSource")

    ms = MemorySource()
    ms.write("hi")
    _print(ms.read())
    ms.close()
    try:
        ms.read()
    except RuntimeError:
        _print("closed")

    rs = ReadOnlySource("snap")
    _print(rs.read())
    try:
        rs.write("x")
    except PermissionError:
        _print("ro")


# ==============================================================================
# Section 10 — Metaclasses
# ==============================================================================
class DocEnforceMeta(type):
    """
    Example 1: Metaclass enforcing class creation rules.

    Rules:
    - Class must have a non-empty docstring.
    - Class must define an integer 'version' attribute.

    Raises:
    - TypeError if a rule is violated during class creation.
    """

    def __new__(mcls, name: str, bases: Tuple[type, ...], ns: Dict[str, Any], **kwargs: Any):
        doc = ns.get("__doc__", "") or ""
        if not doc.strip():
            raise TypeError(f"{name} must have a docstring.")
        if "version" not in ns or not isinstance(ns["version"], int):
            raise TypeError(f"{name} must define an integer class attribute 'version'.")
        return super().__new__(mcls, name, bases, ns, **kwargs)


class ServiceBase(metaclass=DocEnforceMeta):
    """Base enforced service."""
    version: int = 1

    def run(self) -> str:
        return "ok"


class LoggerSingletonMeta(type):
    """
    Example 2: Singleton metaclass — ensures only one instance is created per class.
    """

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in LoggerSingletonMeta._instances:
            LoggerSingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return LoggerSingletonMeta._instances[cls]


class Logger(metaclass=LoggerSingletonMeta):
    """App-wide logger singleton."""
    def __init__(self) -> None:
        self.messages: List[str] = []

    def log(self, msg: str) -> None:
        self.messages.append(msg)

    def dump(self) -> str:
        return "\n".join(self.messages)


def demo_metaclass_1() -> None:
    """
    Input:
        class Good(ServiceBase):
            "Good service."
            version = 2

        print(Good().run())

        try:
            # Dynamically create a bad class missing version/docstring:
            Bad = DocEnforceMeta("Bad", (ServiceBase,), {})
        except TypeError as e:
            print("error")

    Expected output:
        ok
        error
    """
    _print_header("metaclass — enforcement")

    class Good(ServiceBase):
        "Good service."
        version = 2

    _print(Good().run())

    try:
        Bad = DocEnforceMeta("Bad", (ServiceBase,), {})  # noqa: N806
        _print(Bad)  # should not happen
    except TypeError:
        _print("error")


def demo_metaclass_2() -> None:
    """
    Input:
        l1 = Logger()
        l2 = Logger()
        print(l1 is l2)
        l1.log("hello")
        print(l2.dump())

    Expected output:
        True
        hello
    """
    _print_header("metaclass — singleton")

    l1 = Logger()
    l2 = Logger()
    _print(l1 is l2)
    l1.log("hello")
    _print(l2.dump())
    assert l1 is l2 and "hello" in l2.dump()


# ==============================================================================
# Main Orchestration
# ==============================================================================
def main() -> None:
    _print_header("Advanced OOP Masterclass")
    _print_panel(
        "Overview",
        "We will demonstrate:\n"
        "- Constructors (__init__) and Destructors (__del__)\n"
        "- self keyword behavior\n"
        "- Class/Static methods (@classmethod/@staticmethod)\n"
        "- Properties with validation and computed views\n"
        "- Duck typing (behavior over type)\n"
        "- Magic/dunder methods (__str__, __len__, __eq__, __add__, ...)\n"
        "- Abstract Base Classes (ABC)\n"
        "- Metaclasses (class creation control, singletons)\n"
        "\nAll sections include inputs and expected outputs in comments.",
    )

    # Constructors
    demo_constructors()

    # Destructors
    demo_destructors_tracker()
    demo_destructors_safehandle()

    # self
    demo_self_keyword_1()
    demo_self_keyword_2()

    # classmethod
    demo_classmethod_1()
    demo_classmethod_2()

    # staticmethod
    demo_staticmethod_1()
    demo_staticmethod_2()

    # properties
    demo_properties_1()
    demo_properties_2()

    # duck typing
    demo_duck_typing_1()
    demo_duck_typing_2()

    # dunder methods
    demo_dunder_1()
    demo_dunder_2()

    # ABC
    demo_abc_1()
    demo_abc_2()

    # Metaclasses
    demo_metaclass_1()
    demo_metaclass_2()

    _print_panel(
        "Summary",
        "Key takeaways:\n"
        "- Use __init__ for validated construction; prefer context managers over __del__ for resources.\n"
        "- self is just the current instance; explicit and clear.\n"
        "- Class/Static methods separate factory/class-wide logic from instance concerns.\n"
        "- Properties provide Pythonic encapsulation with clean syntax.\n"
        "- Duck typing focuses on capabilities, not type tags.\n"
        "- Dunder methods enable natural syntax; implement them carefully and predictably.\n"
        "- ABCs define contracts; metaclasses can enforce or modify class creation.",
    )


if __name__ == "__main__":
    main()