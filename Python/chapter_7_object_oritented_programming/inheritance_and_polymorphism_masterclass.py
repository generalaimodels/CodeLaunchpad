"""
inheritance_and_polymorphism_masterclass.py — Single-file, deeply commented Python tutorial on:
- Inheritance (Single, Multiple, Multilevel, Hierarchical, Hybrid)
- Polymorphism (Method Overloading, Method Overriding, Operator Overloading)

All explanations live inside this file near the code they explain.
Each demo section includes "Input" and "Expected output" comments.
Uses the 'rich' module for professional console output if available; falls back to plain print.
Toggle the VERBOSE flag to control console verbosity.

Python version: 3.9+ (uses type hints and functools.singledispatchmethod)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Iterable, List, Optional, Sequence, Tuple


# ------------------------------------------------------------------------------
# Configuration: verbosity and optional rich console setup
# ------------------------------------------------------------------------------
VERBOSE: bool = True  # Toggle detailed console output for demos.

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _console = Console()

    def _print_header(title: str) -> None:
        if not VERBOSE:
            return
        _console.rule(Text(title, style="bold cyan"))

    def _print(msg: str) -> None:
        if not VERBOSE:
            return
        _console.print(msg)

    def _print_panel(title: str, body: str) -> None:
        if not VERBOSE:
            return
        _console.print(Panel(body, title=title, border_style="cyan", expand=False))

except Exception:
    _console = None

    def _print_header(title: str) -> None:
        if not VERBOSE:
            return
        print("\n" + "=" * 10 + f" {title} " + "=" * 10)

    def _print(msg: str) -> None:
        if not VERBOSE:
            return
        print(msg)

    def _print_panel(title: str, body: str) -> None:
        if not VERBOSE:
            return
        print("\n" + "=" * 10 + f" {title} " + "=" * 10)
        print(body)


# ==============================================================================
# Section A — Inheritance
# ==============================================================================
# Inheritance lets a child class acquire data (attributes) and behavior (methods)
# from a parent class. It supports:
#  - Code reuse
#  - Specialization via overriding
#  - Polymorphic APIs (same interface, different implementations)
#
# Types covered:
# 1) Single: one parent -> one child
# 2) Multiple: child inherits from multiple parents (mixins / cooperative super)
# 3) Multilevel: grandparent -> parent -> child chain
# 4) Hierarchical: one parent -> many children
# 5) Hybrid: combination of the above (e.g., mixins + classical hierarchy)
# ------------------------------------------------------------------------------


# 1) Single Inheritance ---------------------------------------------------------
class Vehicle:
    """
    Base class to demonstrate single inheritance.
    Attributes:
      - make: manufacturer name (non-empty)
      - model: model name (non-empty)
    Methods:
      - start(), stop(), info(), move()
    """

    def __init__(self, make: str, model: str) -> None:
        if not make or not make.strip():
            raise ValueError("make must be a non-empty string.")
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string.")
        self._make = make.strip()
        self._model = model.strip()
        self._running: bool = False

    def start(self) -> str:
        self._running = True
        return f"{self.info()} started."

    def stop(self) -> str:
        self._running = False
        return f"{self.info()} stopped."

    def info(self) -> str:
        return f"{self._make} {self._model}"

    def move(self) -> str:
        """
        Movement primitive used later by a hybrid inheritance demo.
        """
        return "drive" if self._running else "idle"


class Car(Vehicle):
    """
    Child of Vehicle (single inheritance).
    Extends Vehicle with a 'doors' property and overrides start() to add behavior.
    """

    def __init__(self, make: str, model: str, doors: int) -> None:
        super().__init__(make, model)
        self.doors = doors  # property with validation

    @property
    def doors(self) -> int:
        return self._doors

    @doors.setter
    def doors(self, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("doors must be a positive integer.")
        self._doors = value

    def start(self) -> str:
        """
        Override: refine startup sequence; call super() to keep base semantics.
        """
        base = super().start()
        return f"{base} Seatbelts check OK. Doors={self._doors}."


def demo_single_inheritance() -> None:
    """
    Input:
        c = Car("Tesla", "Model 3", 4)
        print(c.start())
        print(c.move())
        print(c.stop())
        print(isinstance(c, Vehicle), issubclass(Car, Vehicle))

    Expected output:
        Tesla Model 3 started. Seatbelts check OK. Doors=4.
        drive
        Tesla Model 3 stopped.
        True True
    """
    _print_header("Inheritance — Single")

    c = Car("Tesla", "Model 3", 4)
    msg_start = c.start()
    _print(msg_start)
    _print(c.move())
    _print(c.stop())
    _print(f"isinstance(c, Vehicle)={isinstance(c, Vehicle)}")
    _print(f"issubclass(Car, Vehicle)={issubclass(Car, Vehicle)}")

    assert "started" in msg_start and "Doors=4" in msg_start
    assert isinstance(c, Vehicle) and issubclass(Car, Vehicle)


# 2) Multilevel Inheritance -----------------------------------------------------
class Machine:
    """
    Grandparent in a multilevel chain.
    """

    def __init__(self, serial: str) -> None:
        if not serial or not serial.strip():
            raise ValueError("serial must be a non-empty string.")
        self._serial = serial.strip()
        self._powered: bool = False

    def power_on(self) -> str:
        self._powered = True
        return f"Machine[{self._serial}] powered ON."

    def power_off(self) -> str:
        self._powered = False
        return f"Machine[{self._serial}] powered OFF."

    def status(self) -> str:
        return "ON" if self._powered else "OFF"


class Computer(Machine):
    """
    Parent in a multilevel chain: Machine -> Computer -> GamingLaptop
    """

    def boot(self) -> str:
        if not self._powered:
            raise RuntimeError("Cannot boot: power is OFF.")
        return "BIOS -> OS boot sequence complete."


class GamingLaptop(Computer):
    """
    Child in a multilevel chain. Adds GPU turbo mode and overrides boot().
    """

    def __init__(self, serial: str, gpu: str) -> None:
        super().__init__(serial)
        self.gpu = gpu

    def boot(self) -> str:
        """
        Override adds GPU driver initialization, but keeps base boot via super().
        """
        base = super().boot()
        return f"{base} GPU[{self.gpu}] drivers loaded."

    def turbo(self) -> str:
        if not self._powered:
            raise RuntimeError("Turbo requires power ON.")
        return "Turbo mode enabled (fan curve adjusted)."


def demo_multilevel_inheritance() -> None:
    """
    Input:
        gl = GamingLaptop("SN-001", "RTX 4080")
        print(gl.power_on())
        print(gl.boot())
        print(gl.turbo())
        print(gl.power_off())

    Expected output:
        Machine[SN-001] powered ON.
        BIOS -> OS boot sequence complete. GPU[RTX 4080] drivers loaded.
        Turbo mode enabled (fan curve adjusted).
        Machine[SN-001] powered OFF.
    """
    _print_header("Inheritance — Multilevel")

    gl = GamingLaptop("SN-001", "RTX 4080")
    _print(gl.power_on())
    _print(gl.boot())
    _print(gl.turbo())
    _print(gl.power_off())

    assert gl.status() == "OFF"
    try:
        gl.boot()
    except RuntimeError:
        # After power_off, boot should fail
        pass
    else:
        raise AssertionError("Expected RuntimeError after powering OFF.")


# 3) Multiple Inheritance -------------------------------------------------------
class PoweredMixin:
    """
    Mixin for power management. Uses cooperative super() to play well with MRO.
    """

    def __init__(self, *args: Any, power_source: str = "AC", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._power_source = power_source
        self._powered: bool = False

    def power_on(self) -> str:
        self._powered = True
        return f"Power[{self._power_source}] ON."

    def power_off(self) -> str:
        self._powered = False
        return f"Power[{self._power_source}] OFF."

    @property
    def powered(self) -> bool:
        return self._powered


class NetworkedMixin:
    """
    Mixin for network management. Cooperative __init__ to support MRO chains.
    """

    def __init__(self, *args: Any, mac: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mac = mac
        self._connected: bool = False
        self._ip: Optional[str] = None

    def connect(self, ip: str) -> str:
        if not ip:
            raise ValueError("ip must be a non-empty string.")
        self._connected = True
        self._ip = ip
        return f"Connected[{self._mac}] -> {ip}"

    def disconnect(self) -> str:
        self._connected = False
        self._ip = None
        return "Disconnected."

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def ip(self) -> Optional[str]:
        return self._ip


class SmartLight(PoweredMixin, NetworkedMixin):
    """
    Concrete device that is both Powered and Networked.
    Order matters: MRO is SmartLight -> PoweredMixin -> NetworkedMixin -> object
    """

    def __init__(self, name: str, *, mac: str, power_source: str = "AC") -> None:
        # Forward keyword args to the mixins cooperatively
        super().__init__(mac=mac, power_source=power_source)
        self._name = name
        self._brightness: int = 0  # 0..100

    def set_brightness(self, value: int) -> str:
        if not (0 <= value <= 100):
            raise ValueError("brightness must be in [0, 100].")
        if not self.powered or not self.connected:
            raise RuntimeError("Light must be powered and connected before use.")
        self._brightness = value
        return f"{self._name} brightness set to {value}%."

    def identify(self) -> str:
        return f"SmartLight(name={self._name}, mac={self._mac}, power={self._power_source})"


def demo_multiple_inheritance() -> None:
    """
    Input:
        lamp = SmartLight("DeskLamp", mac="AA:BB:CC:DD:EE:FF")
        print(lamp.power_on())
        print(lamp.connect("192.168.1.20"))
        print(lamp.set_brightness(75))
        print(SmartLight.mro())

    Expected output:
        Power[AC] ON.
        Connected[AA:BB:CC:DD:EE:FF] -> 192.168.1.20
        DeskLamp brightness set to 75%.
        [<class '__main__.SmartLight'>, <class '__main__.PoweredMixin'>,
         <class '__main__.NetworkedMixin'>, <class 'object'>]
    """
    _print_header("Inheritance — Multiple")

    lamp = SmartLight("DeskLamp", mac="AA:BB:CC:DD:EE:FF")
    _print(lamp.power_on())
    _print(lamp.connect("192.168.1.20"))
    _print(lamp.set_brightness(75))
    _print(f"MRO: {SmartLight.mro()}")

    assert lamp.powered and lamp.connected and lamp.ip == "192.168.1.20"
    try:
        lamp.set_brightness(120)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for out-of-range brightness.")


# 4) Hierarchical Inheritance ---------------------------------------------------
class Shape:
    """
    Base of a hierarchy. In production, consider abc.ABC; here we raise by default.
    """

    def area(self) -> float:
        raise NotImplementedError("Subclasses must implement area().")

    def perimeter(self) -> float:
        raise NotImplementedError("Subclasses must implement perimeter().")


@dataclass(frozen=True)
class Circle(Shape):
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive.")

    def area(self) -> float:
        return math.pi * (self.radius ** 2)

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius


@dataclass(frozen=True)
class Rectangle(Shape):
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


@dataclass(frozen=True)
class Triangle(Shape):
    base: float
    height: float
    side_a: float
    side_b: float
    side_c: float

    def __post_init__(self) -> None:
        if min(self.base, self.height, self.side_a, self.side_b, self.side_c) <= 0:
            raise ValueError("all sides and dimensions must be positive.")
        # Quick triangle inequality sanity (not full proof for all degeneracies)
        if not (
            self.side_a + self.side_b > self.side_c
            and self.side_a + self.side_c > self.side_b
            and self.side_b + self.side_c > self.side_a
        ):
            raise ValueError("triangle inequality violated.")

    def area(self) -> float:
        return 0.5 * self.base * self.height

    def perimeter(self) -> float:
        return self.side_a + self.side_b + self.side_c


def total_area(shapes: Iterable[Shape]) -> float:
    return sum(s.area() for s in shapes)


def demo_hierarchical_inheritance() -> None:
    """
    Input:
        shapes = [Circle(1.0), Rectangle(2.0, 3.0), Triangle(3.0, 4.0, 3.0, 4.0, 5.0)]
        print([round(s.area(), 2) for s in shapes])
        print(round(total_area(shapes), 2))

    Expected output:
        [3.14, 6.0, 6.0]
        15.14
    """
    _print_header("Inheritance — Hierarchical")

    shapes: List[Shape] = [
        Circle(1.0),
        Rectangle(2.0, 3.0),
        Triangle(3.0, 4.0, 3.0, 4.0, 5.0),
    ]
    areas = [round(s.area(), 2) for s in shapes]
    _print(f"Areas = {areas}")
    _print(f"Total area = {round(total_area(shapes), 2)}")

    assert areas == [3.14, 6.0, 6.0]


# 5) Hybrid Inheritance ---------------------------------------------------------
class Flyable:
    """
    Mixin that composes with Vehicle descendants.
    Cooperative super() to build a movement pipeline.
    """

    def move(self) -> str:  # type: ignore[override]
        return f"{super().move()} -> fly"


class Sailable:
    """
    Mixin that composes with Vehicle descendants.
    """

    def move(self) -> str:  # type: ignore[override]
        return f"{super().move()} -> sail"


class AmphibiousPlaneCar(Flyable, Sailable, Car):
    """
    Hybrid: combines multiple-inheritance mixins with classical Vehicle->Car.
    MRO: AmphibiousPlaneCar -> Flyable -> Sailable -> Car -> Vehicle -> object
    """

    def move(self) -> str:  # type: ignore[override]
        return f"{super().move()} -> amphibious-plane-car"


def demo_hybrid_inheritance() -> None:
    """
    Input:
        apc = AmphibiousPlaneCar("Futura", "X-1", doors=2)
        print(apc.start())
        print(apc.move())

    Expected output:
        Futura X-1 started. Seatbelts check OK. Doors=2.
        drive -> sail -> fly -> amphibious-plane-car
    """
    _print_header("Inheritance — Hybrid")

    apc = AmphibiousPlaneCar("Futura", "X-1", doors=2)
    _print(apc.start())
    _print(apc.move())

    assert apc.move().endswith("amphibious-plane-car")
    # Full chain check (requires started state for 'drive'):
    assert apc.move().startswith("drive")


# ==============================================================================
# Section B — Polymorphism
# ==============================================================================
# Polymorphism = "one name, many forms."
# In Python, polymorphism appears as:
#  - Method Overloading (emulated: defaults, *args/**kwargs, or singledispatchmethod)
#  - Method Overriding (child class redefines a parent method; dynamic dispatch)
#  - Operator Overloading (dunder methods like __add__, __str__, __eq__, __mul__, ...)
# ------------------------------------------------------------------------------


# Method Overloading (Emulated) -------------------------------------------------
class Vec2:
    """
    2D vector to demonstrate method overloading via default args, and operator overloads below.
    """

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    # Emulated overloading: scale(sx[, sy]) — if sy absent, uniform scale.
    def scale(self, sx: float, sy: Optional[float] = None) -> "Vec2":
        """
        Emulated "overloaded" method:
        - scale(k) -> uniform scale by k
        - scale(sx, sy) -> non-uniform scale
        """
        if sy is None:
            return Vec2(self.x * sx, self.y * sx)
        return Vec2(self.x * sx, self.y * sy)

    # Basic operator overloads (used later but fit naturally here)
    def __add__(self, other: Any) -> "Vec2":
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __radd__(self, other: Any) -> "Vec2":
        # Allow sum([...]) with start=0
        if other == 0:
            return self
        if isinstance(other, Vec2):
            return other + self
        return NotImplemented

    def __mul__(self, k: Any) -> "Vec2":
        if isinstance(k, (int, float)):
            return Vec2(self.x * k, self.y * k)
        return NotImplemented

    def __rmul__(self, k: Any) -> "Vec2":
        return self.__mul__(k)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Vec2):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __repr__(self) -> str:
        return f"Vec2(x={self.x:.3f}, y={self.y:.3f})"

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f})"


class Formatter:
    """
    Demonstrates overloading via singledispatchmethod for the first parameter type.
    Not "built-in overloading", but idiomatic and explicit.
    """

    @singledispatchmethod
    def show(self, obj: Any) -> str:
        return f"<unsupported type {type(obj).__name__}>"

    @show.register
    def _(self, obj: int) -> str:
        return f"int: {obj:,}"

    @show.register
    def _(self, obj: float) -> str:
        return f"float: {obj:.3f}"

    @show.register
    def _(self, obj: str) -> str:
        return f"str: {obj!r}"

    @show.register
    def _(self, obj: Sequence) -> str:  # covers list/tuple
        return f"seq(len={len(obj)}): {list(obj)!r}"


def demo_method_overloading() -> None:
    """
    Input:
        v = Vec2(3, 4)
        print(v.scale(2))        # uniform -> (6.00, 8.00)
        print(v.scale(2, 3))     # non-uniform -> (6.00, 12.00)

        f = Formatter()
        print(f.show(42))
        print(f.show(3.14159))
        print(f.show("hello"))
        print(f.show([1, 2, 3]))

    Expected output:
        (6.00, 8.00)
        (6.00, 12.00)
        int: 42
        float: 3.142
        str: 'hello'
        seq(len=3): [1, 2, 3]
    """
    _print_header("Polymorphism — Method Overloading (Emulated)")

    v = Vec2(3, 4)
    _print(str(v.scale(2)))
    _print(str(v.scale(2, 3)))

    f = Formatter()
    _print(f.show(42))
    _print(f.show(3.14159))
    _print(f.show("hello"))
    _print(f.show([1, 2, 3]))

    assert str(v.scale(2)) == "(6.00, 8.00)"
    assert str(v.scale(2, 3)) == "(6.00, 12.00)"
    assert f.show(42).startswith("int:")


# Method Overriding -------------------------------------------------------------
class Animal:
    def speak(self) -> str:
        return "..."  # default fallback


class Dog(Animal):
    def speak(self) -> str:  # override
        return "woof"


class Cat(Animal):
    def speak(self) -> str:  # override
        return "meow"


def chorus(animals: Iterable[Animal]) -> List[str]:
    """
    Polymorphic function: relies on Animal.speak() interface, not concrete types.
    """
    return [a.speak() for a in animals]


def demo_method_overriding() -> None:
    """
    Input:
        pets = [Dog(), Cat(), Dog()]
        print(chorus(pets))
        print([type(p).__name__ + ":" + p.speak() for p in pets])

        # Overriding also shown earlier: Car.start() vs Vehicle.start()

    Expected output:
        ['woof', 'meow', 'woof']
        ['Dog:woof', 'Cat:meow', 'Dog:woof']
    """
    _print_header("Polymorphism — Method Overriding")

    pets = [Dog(), Cat(), Dog()]
    voices = chorus(pets)
    _print(str(voices))
    _print([type(p).__name__ + ":" + p.speak() for p in pets])

    assert voices == ["woof", "meow", "woof"]


# Operator Overloading ----------------------------------------------------------
class Rational:
    """
    Immutable rational number with normalized sign and gcd reduction.
    Demonstrates __add__, __mul__, __eq__, __str__/__repr__, ordering via tuple.
    """

    __slots__ = ("_n", "_d")

    def __init__(self, numerator: int, denominator: int = 1) -> None:
        if denominator == 0:
            raise ZeroDivisionError("denominator cannot be zero.")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise TypeError("numerator and denominator must be integers.")

        # Normalize sign to denominator positive
        sign = -1 if (numerator * denominator) < 0 else 1
        n = abs(numerator)
        d = abs(denominator)

        g = math.gcd(n, d)
        self._n = sign * (n // g)
        self._d = d // g

    @property
    def n(self) -> int:
        return self._n

    @property
    def d(self) -> int:
        return self._d

    def __add__(self, other: Any) -> "Rational":
        if isinstance(other, Rational):
            n = self.n * other.d + other.n * self.d
            d = self.d * other.d
            return Rational(n, d)
        if isinstance(other, int):
            return Rational(self.n + other * self.d, self.d)
        return NotImplemented

    def __radd__(self, other: Any) -> "Rational":
        return self.__add__(other)

    def __mul__(self, other: Any) -> "Rational":
        if isinstance(other, Rational):
            return Rational(self.n * other.n, self.d * other.d)
        if isinstance(other, int):
            return Rational(self.n * other, self.d)
        return NotImplemented

    def __rmul__(self, other: Any) -> "Rational":
        return self.__mul__(other)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Rational):
            return NotImplemented
        return self.n == other.n and self.d == other.d

    def __lt__(self, other: "Rational") -> bool:
        if not isinstance(other, Rational):
            return NotImplemented
        return self.n * other.d < other.n * self.d

    def __repr__(self) -> str:
        return f"Rational({self.n}, {self.d})"

    def __str__(self) -> str:
        return f"{self.n}/{self.d}"


def demo_operator_overloading() -> None:
    """
    Input:
        r1 = Rational(1, 2)
        r2 = Rational(3, 4)
        print(r1 + r2)   # 5/4
        print(2 + r1)    # 5/2
        print(r1 * r2)   # 3/8
        print(3 * r1)    # 3/2
        v1 = Vec2(1, 2)
        v2 = Vec2(3, 5)
        print(v1 + v2)   # (4.00, 7.00)
        print(2 * v1)    # (2.00, 4.00)

    Expected output:
        5/4
        5/2
        3/8
        3/2
        (4.00, 7.00)
        (2.00, 4.00)
    """
    _print_header("Polymorphism — Operator Overloading")

    r1 = Rational(1, 2)
    r2 = Rational(3, 4)
    _print(str(r1 + r2))
    _print(str(2 + r1))
    _print(str(r1 * r2))
    _print(str(3 * r1))

    v1, v2 = Vec2(1, 2), Vec2(3, 5)
    _print(str(v1 + v2))
    _print(str(2 * v1))

    assert str(r1 + r2) == "5/4"
    assert str(2 + r1) == "5/2"
    assert str(v1 + v2) == "(4.00, 7.00)"


# ==============================================================================
# Additional micro-examples and utilities
# ==============================================================================
def demo_sum_vec2() -> None:
    """
    Demonstrates sum() with __radd__ support.
    Input:
        vectors = [Vec2(1, 1), Vec2(2, 3), Vec2(-1, 4)]
        print(sum(vectors))  # thanks to __radd__ with 0 handling

    Expected output:
        (2.00, 8.00)
    """
    _print_header("Operator Overloading — sum(Vec2)")

    vectors = [Vec2(1, 1), Vec2(2, 3), Vec2(-1, 4)]
    total = sum(vectors)  # starts from 0; our __radd__ handles (0 + Vec2)
    _print(str(total))
    assert total == Vec2(2, 8)


def demo_mro_inspection() -> None:
    """
    Quick MRO peek for hybrid class to visualize resolution order.
    """
    _print_header("MRO — AmphibiousPlaneCar")
    _print(str(AmphibiousPlaneCar.mro()))


# ==============================================================================
# Main Orchestration
# ==============================================================================
def main() -> None:
    _print_header("OOP Masterclass — Inheritance & Polymorphism")
    _print_panel(
        "Overview",
        "Sections:\n"
        "- Inheritance: Single, Multiple, Multilevel, Hierarchical, Hybrid\n"
        "- Polymorphism: Overloading (emulated), Overriding, Operator overloading\n\n"
        "Each demo prints example Input/Expected behavior, with invariants guarded by exceptions.",
    )

    # Inheritance demos (at least two examples; we provide five)
    demo_single_inheritance()
    demo_multilevel_inheritance()
    demo_multiple_inheritance()
    demo_hierarchical_inheritance()
    demo_hybrid_inheritance()

    # Polymorphism demos (at least two examples; we provide three)
    demo_method_overloading()
    demo_method_overriding()
    demo_operator_overloading()

    # Extras
    demo_sum_vec2()
    demo_mro_inspection()

    _print_panel(
        "Summary",
        "Key points:\n"
        "- Inheritance enables reuse and specialization; choose simple hierarchies.\n"
        "- Multiple inheritance works with cooperative super() and mixins.\n"
        "- Polymorphism: same API, different behaviors; prefer clear contracts.\n"
        "- Operator overloads should be intuitive and type-safe.",
    )


if __name__ == "__main__":
    main()