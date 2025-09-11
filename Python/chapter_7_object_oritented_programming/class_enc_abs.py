"""
o_oop_masterclass.py  —  Single-file, deeply commented Python tutorial on:
- Class
- Object
- Encapsulation
- Abstraction (via ABC: Abstract Base Classes)

This file is intentionally verbose (see VERBOSE flag) and self-contained:
- All explanations and rationale live in comments/docstrings near the code they explain.
- Each demo section includes "Input" and "Expected output" comments.
- No external text is required beyond this file.
- Uses the 'rich' module for nicer console output if available; falls back to plain print.

--------------------------------------------------------------------------------
Core concepts
--------------------------------------------------------------------------------
1) Class
   - A class is a blueprint for creating objects. It defines data (attributes) and
     behavior (methods). Class attributes are shared by all instances;
     instance attributes belong to specific objects.

2) Object
   - An object is an instance of a class. It encapsulates state (attribute values)
     and exposes behavior (methods). Objects have identity (is), type, and value.

3) Encapsulation
   - Encapsulation wraps data and methods inside a class to enforce invariants and
     hide internal representation. In Python:
       - A single leading underscore (e.g., _status) means "protected" by convention:
         “internal use” (not enforced).
       - A double leading underscore (e.g., __balance) triggers name-mangling to
         reduce accidental access (still possible, but strongly discouraged).
       - Properties (@property) provide controlled access (getters/setters/validators)
         while retaining attribute-like syntax.

4) Abstraction (ABC)
   - Abstraction hides implementation details and exposes essential behavior.
   - The abc module lets you define abstract classes with @abstractmethod (and
     abstract @property). Abstract classes cannot be instantiated directly.
   - Users program to the interface (the abstract class), not a concrete type.

--------------------------------------------------------------------------------
Running this file
--------------------------------------------------------------------------------
- Just run the module directly with Python 3.11+ (type hints use | and typing):
    python o_oop_masterclass.py
- If 'rich' is installed, output will be nicely formatted; otherwise, it's plain.

"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional


# ------------------------------------------------------------------------------
# Configuration: verbosity and optional rich console setup
# ------------------------------------------------------------------------------
VERBOSE: bool = True  # Toggle detailed console output for demos.

try:
    # Rich is optional. If unavailable, we degrade gracefully to print().
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


# ------------------------------------------------------------------------------
# Section 1 — Class & Object Basics
# ------------------------------------------------------------------------------
class Vector2D:
    """
    Vector2D: Minimal, clean example of a class with:
    - Class attribute (dimensions)
    - Instance attributes (x, y)
    - Instance methods (magnitude, translate)
    - Classmethod (factory constructor)
    - Staticmethod (utility not tied to instance or class state)
    - Dunder methods (__repr__, __eq__) for professional UX in debugging/comparison

    Motivations:
    - Demonstrate the blueprint nature of classes.
    - Distinguish class attributes from instance attributes.
    - Show object identity vs. equality.
    - Show clean, well-typed API with validations and computed behavior.

    Invariants (documented and enforced where reasonable):
    - x and y are floats (we coerce to float for safety).
    """

    # Class attribute: shared by all instances (read-only by convention).
    dimensions: int = 2

    def __init__(self, x: float, y: float) -> None:
        self.x: float = float(x)
        self.y: float = float(y)

    def magnitude(self) -> float:
        """
        Returns the Euclidean length (L2 norm).
        """
        return math.hypot(self.x, self.y)

    def translate(self, dx: float, dy: float) -> "Vector2D":
        """
        Pure method that returns a new vector; original remains unchanged.
        """
        return Vector2D(self.x + dx, self.y + dy)

    @classmethod
    def from_iterable(cls, data: Iterable[float]) -> "Vector2D":
        """
        Factory constructor. Accepts any iterable of two numbers.
        """
        xs = tuple(float(v) for v in data)
        if len(xs) != 2:
            raise ValueError("Vector2D.from_iterable expects exactly two numbers.")
        return cls(xs[0], xs[1])

    @staticmethod
    def dot(a: "Vector2D", b: "Vector2D") -> float:
        """
        Dot product between two Vector2D instances.
        """
        if not isinstance(a, Vector2D) or not isinstance(b, Vector2D):
            raise TypeError("dot expects two Vector2D instances.")
        return a.x * b.x + a.y * b.y

    def __repr__(self) -> str:
        return f"Vector2D(x={self.x:.3f}, y={self.y:.3f})"

    def __eq__(self, other: Any) -> bool:
        """
        Professional equality for floats: tolerance-aware.
        """
        if not isinstance(other, Vector2D):
            return NotImplemented
        return (
            math.isclose(self.x, other.x, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(self.y, other.y, rel_tol=1e-9, abs_tol=1e-12)
        )


def demo_class_and_object_basics() -> None:
    """
    Demonstrates:
    - Declaring objects (instances) from a class
    - Identity vs equality
    - Class attribute vs instance attributes
    - Methods, classmethod, staticmethod

    Input:
        v = Vector2D(3, 4)
        print(v.magnitude())
        print(Vector2D.dimensions)
        v2 = v.translate(1, -1)
        print(v2)
        print(Vector2D.dot(v, v2))
        print(v == Vector2D(3, 4))
        print(v is Vector2D(3, 4))  # identity check

    Expected output:
        5.0
        2
        Vector2D(x=4.000, y=3.000)
        24.0
        True
        False
    """
    _print_header("Class & Object — Vector2D Basics")

    v = Vector2D(3, 4)
    _print(f"v = {v} | magnitude = {v.magnitude()}")  # 5.0 expected

    # Class attribute: the same for all instances.
    _print(f"Vector2D.dimensions (class attribute) = {Vector2D.dimensions}")  # 2

    v2 = v.translate(1, -1)
    _print(f"v2 (translated) = {v2}")  # Vector2D(x=4.000, y=3.000)

    dp = Vector2D.dot(v, v2)
    _print(f"dot(v, v2) = {dp}")  # 24.0

    _print(f"v == Vector2D(3, 4) ? {v == Vector2D(3, 4)}")  # True
    _print(f"v is Vector2D(3, 4) ? {v is Vector2D(3, 4)}")  # False (different object)

    # Identity demonstration:
    alias = v
    _print(f"alias is v ? {alias is v}")  # True, both names point to the same object

    # Assertions (silent if correct) for self-checking:
    assert math.isclose(v.magnitude(), 5.0)
    assert Vector2D.dimensions == 2
    assert repr(v2) == "Vector2D(x=4.000, y=3.000)"
    assert math.isclose(dp, 24.0)
    assert v == Vector2D(3, 4)
    assert v is not Vector2D(3, 4)
    assert alias is v


# ------------------------------------------------------------------------------
# Section 2 — Encapsulation
# ------------------------------------------------------------------------------
class BankAccount:
    """
    BankAccount demonstrates encapsulation:
    - Protected attribute by convention: _status
    - Private attribute via name-mangling: __balance
    - Property for read-only exposure of balance
    - Methods enforce invariants (no negative deposit/withdraw)
    - Clean API for controlled mutation

    Notes on encapsulation in Python:
    - Single underscore prefix (e.g., _status) signals "internal use"; not enforced.
    - Double underscore prefix (e.g., __balance) triggers name-mangling to _Class__attr,
      making accidental access harder. It's not a strict privacy model, but it discourages misuse.
    - The idiomatic way to protect invariants is to keep attributes non-public and expose
      controlled access via properties and methods.

    Exceptions:
    - ValueError for invalid inputs (negative amounts, overdraft).
    - RuntimeError if operations not allowed due to status.
    """

    _id_counter: int = 1000  # class attribute to assign simple unique IDs

    def __init__(self, owner: str, account_number: str, opening_balance: float = 0.0) -> None:
        if not owner or not owner.strip():
            raise ValueError("owner must be a non-empty string.")
        if not account_number or not account_number.strip():
            raise ValueError("account_number must be a non-empty string.")
        if opening_balance < 0:
            raise ValueError("opening_balance cannot be negative.")

        self._status: str = "ACTIVE"  # protected by convention: internal state machine
        self._owner: str = owner.strip()
        self._account_number: str = account_number.strip()
        self._id: int = self._next_id()
        self.__balance: float = float(opening_balance)  # private via name-mangling

    # ------------------------------
    # Private helpers and internals
    # ------------------------------
    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def _ensure_active(self) -> None:
        if self._status != "ACTIVE":
            raise RuntimeError(f"Operation not allowed while status={self._status}.")

    # ------------------------------
    # Properties (encapsulated access)
    # ------------------------------
    @property
    def id(self) -> int:
        """Read-only numeric identifier."""
        return self._id

    @property
    def owner(self) -> str:
        return self._owner

    @owner.setter
    def owner(self, value: str) -> None:
        if not value or not value.strip():
            raise ValueError("owner cannot be empty.")
        self._owner = value.strip()

    @property
    def account_number(self) -> str:
        return self._account_number

    @property
    def status(self) -> str:
        """Exposed as read-only property; modifications go through methods."""
        return self._status

    @property
    def balance(self) -> float:
        """
        Read-only property: external callers can see the balance but cannot assign it.
        Mutation is funneled through deposit/withdraw/transfer with validation.
        """
        return self.__balance

    # ------------------------------
    # Public API (validated operations)
    # ------------------------------
    def deposit(self, amount: float) -> float:
        """
        Deposit amount > 0. Returns new balance.
        """
        self._ensure_active()
        if amount <= 0:
            raise ValueError("deposit amount must be positive.")
        self.__balance += float(amount)
        return self.__balance

    def withdraw(self, amount: float) -> float:
        """
        Withdraw amount > 0 not exceeding current balance. Returns new balance.
        """
        self._ensure_active()
        if amount <= 0:
            raise ValueError("withdraw amount must be positive.")
        if amount > self.__balance:
            raise ValueError("insufficient funds.")
        self.__balance -= float(amount)
        return self.__balance

    def transfer(self, to: "BankAccount", amount: float) -> None:
        """
        Atomic-ish transfer (simple example; no concurrency here):
        - Withdraw from self, deposit into 'to'.

        Raises:
        - ValueError if amount invalid or insufficient funds.
        - RuntimeError if either account is not in ACTIVE status.
        """
        if not isinstance(to, BankAccount):
            raise TypeError("to must be a BankAccount.")
        self._ensure_active()
        to._ensure_active()
        self.withdraw(amount)
        to.deposit(amount)

    def freeze(self) -> None:
        """Set account status to FROZEN."""
        self._status = "FROZEN"

    def unfreeze(self) -> None:
        """Set account status to ACTIVE."""
        self._status = "ACTIVE"

    def __repr__(self) -> str:
        # Include essential, non-sensitive info. Avoid printing private fields in real systems.
        return (
            f"BankAccount(id={self._id}, owner='{self._owner}', "
            f"acct='{self._account_number}', status='{self._status}')"
        )


class PremiumAccount(BankAccount):
    """
    Inherits from BankAccount to demonstrate "protected" attribute use in subclasses.
    - Adds overdraft_limit (encapsulated via property with validation).
    - Uses _status (protected by convention) to allow certain premium operations.

    Note:
    - The single underscore (_status) is not enforced privacy; it's a convention telling
      consumers "this is internal; don't touch it".
    """

    def __init__(self, owner: str, account_number: str, opening_balance: float = 0.0) -> None:
        super().__init__(owner, account_number, opening_balance)
        self._overdraft_limit: float = 0.0  # protected by convention

    @property
    def overdraft_limit(self) -> float:
        return self._overdraft_limit

    @overdraft_limit.setter
    def overdraft_limit(self, value: float) -> None:
        if value < 0:
            raise ValueError("overdraft_limit cannot be negative.")
        self._overdraft_limit = float(value)

    def withdraw(self, amount: float) -> float:
        """
        Override to allow overdraft up to overdraft_limit.
        """
        self._ensure_active()
        if amount <= 0:
            raise ValueError("withdraw amount must be positive.")
        if amount > (self.balance + self._overdraft_limit):
            raise ValueError("insufficient funds (beyond overdraft limit).")
        # Accessing private balance from subclass? Can't directly use __balance;
        # we must route through public API since __balance is name-mangled in base.
        # We'll implement using base public operations:
        if amount <= self.balance:
            return super().withdraw(amount)
        # Overdraft: drain to zero, then represent negative via an internal method
        # For demonstration, we'll simulate by temporarily depositing negative:
        deficit = amount - self.balance  # positive
        # This two-step approach demonstrates respecting encapsulation:
        super().withdraw(self.balance)  # brings to zero
        # emulate negative display by storing IOU as a "shadow" via overdraft usage.
        # In real implementations, balance might allow negatives, but our base forbids it.
        # We'll adapt by using a private extension (careful and explicit).
        self._apply_overdraft(deficit)
        return self.balance  # remains 0.0 in this simplified example

    def _apply_overdraft(self, used: float) -> None:
        """
        Internal helper to demonstrate subclass internals. In a real system, the base
        class would be designed to allow negative balances or have hooks for overdraft.
        Here, we'll just reduce the overdraft_limit to reflect usage.
        """
        self._overdraft_limit -= used
        if self._overdraft_limit < 0:
            # Defensive: should not occur due to previous validation.
            self._overdraft_limit = 0.0


def demo_encapsulation() -> None:
    """
    Demonstrates:
    - Protected _status, private __balance (name-mangling)
    - Enforced invariants through methods and properties
    - Attempting direct access to private attribute
    - Subclass usage of protected members

    Input:
        a = BankAccount("Alice", "AC-001", 100.0)
        a.deposit(50)
        a.withdraw(20)
        print(a.balance)
        try: print(a.__balance)  # AttributeError
        except AttributeError: print("no access")

        p = PremiumAccount("Bob", "PR-777", 200.0)
        p.overdraft_limit = 150.0
        p.withdraw(300.0)  # overdraft usage
        print(p.overdraft_limit)

        a.freeze()
        try: a.deposit(10)  # RuntimeError due to frozen status
        except RuntimeError: print("frozen")

    Expected output (values may vary slightly based on intermediate representation):
        130.0
        no access
        50.0
        frozen
    """
    _print_header("Encapsulation — BankAccount & PremiumAccount")

    a = BankAccount("Alice", "AC-001", opening_balance=100.0)
    _print(f"Created account: {a}")
    _print(f"Initial balance (read-only): {a.balance:.2f}")

    new_balance = a.deposit(50.0)
    _print(f"After deposit(50): {new_balance:.2f}")
    new_balance = a.withdraw(20.0)
    _print(f"After withdraw(20): {new_balance:.2f}")  # Expected 130.0

    try:
        # This will raise AttributeError because __balance is name-mangled.
        getattr(a, "__balance")  # noqa: B009 (intentional for demonstration)
    except AttributeError as exc:
        _print("Accessing a.__balance -> AttributeError (expected)")

    # WARNING: Name-mangling means the private name exists as _BankAccount__balance.
    # You CAN access it, but you SHOULD NOT. This defeats encapsulation and may break
    # invariants. It's shown here for completeness only.
    mangled_value = getattr(a, "_BankAccount__balance")
    _print(f"(Do not do this) Mangled private access: _BankAccount__balance = {mangled_value:.2f}")

    # Demonstrate subclass with protected attribute (_status) and overdraft logic:
    p = PremiumAccount("Bob", "PR-777", opening_balance=200.0)
    p.overdraft_limit = 150.0
    _print(f"\nCreated premium account: {p}")
    _print(f"Initial balance: {p.balance:.2f}, overdraft_limit: {p.overdraft_limit:.2f}")

    p.withdraw(300.0)  # Uses overdraft (simplified demo semantics)
    _print(f"After withdraw(300): balance={p.balance:.2f}, overdraft_limit={p.overdraft_limit:.2f}")

    # Freezing prevents operations:
    a.freeze()
    _print(f"\nAccount {a.id} frozen. status={a.status}")
    try:
        a.deposit(10.0)
    except RuntimeError:
        _print("Deposit rejected while frozen (expected).")

    # Assertions (silent if correct):
    assert math.isclose(a.balance, 130.0)
    assert p.overdraft_limit <= 150.0
    assert a.status == "FROZEN"


# ------------------------------------------------------------------------------
# Section 3 — Abstraction via ABC (Abstract Base Classes)
# ------------------------------------------------------------------------------
class PaymentProcessor(ABC):
    """
    Abstract interface describing a payment processor.
    Abstraction: consumers depend on this interface, not concrete implementations.

    Key points:
    - @abstractmethod methods must be implemented by subclasses; otherwise they are abstract.
    - @property + @abstractmethod for abstract properties.
    - Default (non-abstract) methods provide reusable behavior.

    Contract:
    - process(amount, currency) -> transaction_id (str)
    - refund(transaction_id) -> bool
    - provider (property) -> str
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Human-readable provider name."""
        raise NotImplementedError

    @abstractmethod
    def process(self, amount: float, currency: str = "USD") -> str:
        """
        Charge the given amount in the given currency.
        Returns a transaction id string.
        """
        raise NotImplementedError

    @abstractmethod
    def refund(self, transaction_id: str) -> bool:
        """
        Refund the given transaction id.
        Returns True if successful, False otherwise.
        """
        raise NotImplementedError

    # Default reusable behavior: validation helper.
    def _validate_amount(self, amount: float) -> None:
        if not isinstance(amount, (int, float)):
            raise TypeError("amount must be a number.")
        if amount <= 0:
            raise ValueError("amount must be positive.")
        if math.isinf(amount) or math.isnan(amount):
            raise ValueError("amount must be a finite number.")

    # Default reusable behavior: currency normalization.
    def _normalize_currency(self, currency: str) -> str:
        if not isinstance(currency, str) or not currency:
            raise ValueError("currency must be a non-empty string.")
        return currency.upper()


@dataclass
class StripeProcessor(PaymentProcessor):
    """
    Concrete implementation of PaymentProcessor.
    Implementation details are hidden behind the abstract interface.

    Attributes:
    - api_key: fake credential for demo purposes.

    Notes:
    - The consumer (caller) doesn't care how StripeProcessor performs a charge;
      they only rely on process() and refund() per the abstract contract.
    """

    api_key: str

    @property
    def provider(self) -> str:
        return "Stripe"

    def process(self, amount: float, currency: str = "USD") -> str:
        self._validate_amount(amount)
        currency = self._normalize_currency(currency)
        # Simulate processing and returning a transaction id:
        return f"stripe_txn_{abs(hash((amount, currency, self.api_key))) % 10_000_000}"

    def refund(self, transaction_id: str) -> bool:
        # Simulate a partial success rule for demo:
        return transaction_id.startswith("stripe_txn_")


@dataclass
class PayPalProcessor(PaymentProcessor):
    """
    Another concrete implementation with a different internal approach.
    """

    client_id: str
    secret: str

    @property
    def provider(self) -> str:
        return "PayPal"

    def process(self, amount: float, currency: str = "USD") -> str:
        self._validate_amount(amount)
        currency = self._normalize_currency(currency)
        # Different token scheme to show polymorphism:
        return f"paypal_txn_{abs(hash((self.client_id, amount, currency))) % 10_000_000}"

    def refund(self, transaction_id: str) -> bool:
        # Simulated rule: pretend only even IDs are refundable
        try:
            tail = int(transaction_id.rsplit("_", 1)[-1])
            return transaction_id.startswith("paypal_txn_") and (tail % 2 == 0)
        except Exception:
            return False


def handle_checkout(processor: PaymentProcessor, amount: float, currency: str = "USD") -> str:
    """
    Function depends on the abstract interface only (abstraction in action).

    Returns a human-readable message including a transaction id.
    """
    txn = processor.process(amount, currency)
    return f"{processor.provider} processed {amount:.2f} {currency}; txn={txn}"


def demo_abstraction() -> None:
    """
    Demonstrates:
    - ABC cannot be instantiated directly.
    - Concrete classes implement the abstract interface.
    - Polymorphism: same call site works for different implementations.

    Input:
        try:
            PaymentProcessor()
        except TypeError:
            print("cannot instantiate")

        stripe = StripeProcessor(api_key="sk_test_123")
        msg1 = handle_checkout(stripe, 29.99, "usd")
        print(msg1)
        ok1 = stripe.refund(msg1.split("txn=")[1])
        print(ok1)

        paypal = PayPalProcessor(client_id="id_abc", secret="s3cr3t")
        msg2 = handle_checkout(paypal, 49.50, "eur")
        print(msg2)
        ok2 = paypal.refund(msg2.split("txn=")[1])
        print(ok2)

    Expected output (transaction IDs vary due to hashing):
        cannot instantiate
        Stripe processed 29.99 USD; txn=stripe_txn_<digits>
        True
        PayPal processed 49.50 EUR; txn=paypal_txn_<digits>
        True or False (based on simulated rule)
    """
    _print_header("Abstraction — PaymentProcessor (ABC)")

    # ABC instantiation should fail:
    try:
        PaymentProcessor()  # type: ignore[abstract]
    except TypeError:
        _print("Cannot instantiate abstract PaymentProcessor (expected).")

    stripe = StripeProcessor(api_key="sk_test_123")
    msg1 = handle_checkout(stripe, 29.99, "usd")
    _print(msg1)
    txn1 = msg1.split("txn=")[1]
    ok1 = stripe.refund(txn1)
    _print(f"Stripe refund({txn1}) -> {ok1}")

    paypal = PayPalProcessor(client_id="id_abc", secret="s3cr3t")
    msg2 = handle_checkout(paypal, 49.50, "eur")
    _print(msg2)
    txn2 = msg2.split("txn=")[1]
    ok2 = paypal.refund(txn2)
    _print(f"PayPal refund({txn2}) -> {ok2}")

    # Assertions are not deterministic for refund results due to demo rules; skip strict checks.


# ------------------------------------------------------------------------------
# Additional micro-examples to cement concepts
# ------------------------------------------------------------------------------
class ReadOnlyExample:
    """
    Demonstrates a read-only computed property with no backing attribute exposed.

    Use-case:
    - Expose a value derived from internal state without leaking internals.
    """

    def __init__(self, data: Iterable[float]) -> None:
        self._values = tuple(float(x) for x in data)

    @property
    def average(self) -> float:
        if not self._values:
            return float("nan")
        return sum(self._values) / len(self._values)

    def __repr__(self) -> str:
        return f"ReadOnlyExample(values={self._values}, average={self.average:.3f})"


def demo_read_only_property() -> None:
    """
    Input:
        r = ReadOnlyExample([1, 2, 3, 4])
        print(r.average)
        try: r.average = 10
        except AttributeError: print("read-only")

    Expected output:
        2.5
        read-only
    """
    _print_header("Encapsulation — Read-only Property")
    r = ReadOnlyExample([1, 2, 3, 4])
    _print(f"r = {r}")
    _print(f"r.average = {r.average}")

    try:
        # Attempting to assign to a read-only @property raises AttributeError:
        setattr(r, "average", 10)  # noqa: B010 (intentional)
    except AttributeError:
        _print("Attempt to set read-only property -> AttributeError (expected).")


# ------------------------------------------------------------------------------
# Main execution with structured demonstration
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Orchestrates all demos. Each demo prints its own section title and outputs.
    """
    _print_header("OOP Masterclass — Class, Object, Encapsulation, Abstraction")
    _print_panel(
        "Overview",
        "This run will demonstrate:\n"
        "- Class/Object basics with Vector2D\n"
        "- Encapsulation with BankAccount/PremiumAccount\n"
        "- Abstraction via PaymentProcessor (ABC)\n"
        "- Read-only property example\n\n"
        "Toggle VERBOSE at the top of the file to silence or show detailed output.",
    )

    demo_class_and_object_basics()
    demo_encapsulation()
    demo_abstraction()
    demo_read_only_property()

    _print_panel("Summary",
                 "Key takeaways:\n"
                 "- Class defines data+behavior. Objects are instances with identity.\n"
                 "- Encapsulation: prefer non-public attributes + properties/methods.\n"
                 "- Abstraction (ABC): code to interfaces; hide implementation details.\n"
                 "- Validate inputs; raise precise exceptions to protect invariants.")


if __name__ == "__main__":
    main()