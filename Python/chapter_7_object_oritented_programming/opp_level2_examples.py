# Advanced OOP Examples Level 2
# Enhancing understanding of advanced OOP concepts in Python 🐍

# Example 1: Using Metaclasses to Control Class Creation
class Meta(type):
    # 🛠️ Metaclass controlling class creation
    def __new__(cls, name, bases, attrs):
        print(f"Creating class {name}")
        attrs['id'] = '12345'  # Adding an attribute to the class
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=Meta):
    # 🎭 Class using custom metaclass
    def method(self):
        pass

print(MyClass.id)  # Outputs: 12345

# Example 2: Singleton Pattern using Metaclass
class Singleton(type):
    # 🗝️ Metaclass for Singleton pattern
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # 🆕 Creating new instance if not exists
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MySingleton(metaclass=Singleton):
    # 🏛️ Singleton class
    pass

obj1 = MySingleton()
obj2 = MySingleton()
print(obj1 is obj2)  # Outputs: True

# Example 3: Custom __new__ Method for Immutable Classes
class ImmutablePoint:
    # 📍 Immutable point class
    def __new__(cls, x, y):
        # 🎩 Custom instance creation
        instance = super().__new__(cls)
        instance._x = x
        instance._y = y
        return instance

    def __setattr__(self, key, value):
        # 🚫 Prevent attribute modification
        raise AttributeError("Cannot modify immutable instance")

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

point = ImmutablePoint(3, 4)
print(f"Point coordinates: ({point.x}, {point.y})")
# point.x = 5  # ❌ Would raise AttributeError

# Example 4: Abstract Base Classes with abc Module
from abc import ABC, abstractmethod

class Strategy(ABC):
    # ♟️ Abstract base class for strategy pattern
    @abstractmethod
    def execute(self, data):
        pass

class ConcreteStrategyA(Strategy):
    # 🅰️ Implementation of Strategy A
    def execute(self, data):
        print(f"Strategy A processing {data}")

class ConcreteStrategyB(Strategy):
    # 🅱️ Implementation of Strategy B
    def execute(self, data):
        print(f"Strategy B processing {data}")

strategy = ConcreteStrategyA()
strategy.execute("Sample Data")

# Example 5: Property Decorators for Managed Attributes
class Celsius:
    # 🌡️ Class to represent temperature in Celsius
    def __init__(self, temperature=0):
        self.temperature = temperature  # Initiates the setter

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    @property
    def temperature(self):
        # 👀 Getter method
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        # 🔧 Setter method with validation
        if value < -273.15:
            raise ValueError("Temperature below -273.15°C is not possible!")
        self._temperature = value

temp = Celsius(25)
print(f"{temp.temperature}°C is {temp.to_fahrenheit()}°F")
# temp.temperature = -300  # ❌ Raises ValueError

# Example 6: Operator Overloading with Magic Methods
class Vector:
    # 🧭 Vector class with operator overloading
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        # ➕ Overloading addition operator
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        # ✖️ Overloading multiplication with scalar
        return Vector(self.x * scalar, self.y * scalar)

    def __repr__(self):
        # 🖨️ Official string representation
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(4, 5)
print(v1 + v2)       # Outputs: Vector(6, 8)
print(v1 * 3)        # Outputs: Vector(6, 9)

# Example 7: Context Managers with __enter__ and __exit__
class FileManager:
    # 📂 Context manager for file operations
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        # 🚪 Entering context
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        # 🚪 Exiting context
        self.file.close()

with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')  # File is automatically closed after this block

# Example 8: Decorators in Classes
def method_decorator(method):
    # 🎀 Decorator for methods
    def wrapper(self, *args, **kwargs):
        print(f"Calling method {method.__name__}")
        return method(self, *args, **kwargs)
    return wrapper

class MyClass:
    # 🏷️ Class with decorated method
    @method_decorator
    def display(self):
        print("Display method called")

obj = MyClass()
obj.display()
# Outputs:
# Calling method display
# Display method called

# Example 9: Class Decorators
def class_decorator(cls):
    # 🎁 Decorator for classes
    cls.category = 'Decorated Class'  # Adding attribute to class
    return cls

@class_decorator
class DecoratedClass:
    # 🎨 Class being decorated
    pass

print(DecoratedClass.category)  # Outputs: Decorated Class

# Example 10: Multiple Inheritance with Method Resolution Order (MRO)
class A:
    def method(self):
        print("Method from class A")

class B(A):
    def method(self):
        print("Method from class B")
        super().method()

class C(A):
    def method(self):
        print("Method from class C")
        super().method()

class D(B, C):
    def method(self):
        print("Method from class D")
        super().method()

d = D()
d.method()
# Outputs:
# Method from class D
# Method from class B
# Method from class C
# Method from class A

# Example 11: Using Slots to Reduce Memory Usage
class Point:
    # 📌 Class using slots to prevent dynamic attribute addition
    __slots__ = ('x', 'y')  # Only these attributes are allowed

    def __init__(self, x, y):
        self.x = x
        self.y = y

point = Point(1, 2)
# point.z = 3  # ❌ AttributeError: 'Point' object has no attribute 'z'

# Example 12: Overriding __call__ Method
class Counter:
    # 🔢 Class instances act like callable objects
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        print(f"Called {self.count} times")

counter = Counter()
counter()  # Outputs: Called 1 times
counter()  # Outputs: Called 2 times

# Example 13: Implementing the Observer Pattern
class Subject:
    # 👁️ Subject being observed
    def __init__(self):
        self._observers = []

    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

    def attach(self, observer):
        self._observers.append(observer)

class Observer:
    # 🕵️ Observer receiving updates
    def update(self, message):
        print(f"Observer received: {message}")

subject = Subject()
observer1 = Observer()
observer2 = Observer()
subject.attach(observer1)
subject.attach(observer2)
subject.notify("An event has occurred")

# Example 14: Using Classmethods and Staticmethods
class Calculator:
    # 🔢 Calculator class
    @staticmethod
    def add(a, b):
        # ➕ Static method doesn't access instance or class
        return a + b

    @classmethod
    def multiply(cls, a, b):
        # ✖️ Class method accesses the class
        return a * b

print(Calculator.add(5, 7))          # Outputs: 12
print(Calculator.multiply(5, 7))     # Outputs: 35

# Example 15: Creating Read-Only Properties
class ReadOnly:
    # 🔒 Class with read-only property
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

obj = ReadOnly(10)
print(obj.value)  # Outputs: 10
# obj.value = 20   # ❌ AttributeError: can't set attribute

# Example 16: Fluent Interface with Method Chaining
class Builder:
    # 🧱 Builder class with method chaining
    def __init__(self):
        self.product = ''

    def add_part(self, part):
        self.product += part
        return self  # 🔗 Returning self for chaining

    def build(self):
        return self.product

builder = Builder()
product = builder.add_part('PartA ').add_part('PartB ').build()
print(product)  # Outputs: PartA PartB

# Example 17: Weak References to Avoid Reference Cycles
import weakref

class Node:
    # 🌳 Node class with weak reference
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def set_parent(self, node):
        self.parent = weakref.ref(node)  # 🧷 Weak reference to parent

root = Node('root')
child = Node('child')
child.set_parent(root)
print(child.parent().value)  # Outputs: root

# Example 18: Custom Containers with __getitem__, __setitem__
class Squares:
    # 🔢 Class behaving like a list of squares
    def __getitem__(self, index):
        return index ** 2

squares = Squares()
print(squares[5])  # Outputs: 25
for i in range(1, 4):
    print(squares[i])  # Outputs: 1, 4, 9

# Example 19: Method Dispatch with functools.singledispatch
from functools import singledispatch

@singledispatch
def process(arg):
    print(f"Default processing for {arg}")

@process.register(int)
def _(arg):
    print(f"Processing integer: {arg}")

@process.register(str)
def _(arg):
    print(f"Processing string: {arg}")

process(10)       # Outputs: Processing integer: 10
process("hello")  # Outputs: Processing string: hello
process([1, 2, 3])# Outputs: Default processing for [1, 2, 3]

# Example 20: Chain of Responsibility Pattern
class Handler:
    # 🔗 Base handler class
    def __init__(self, successor=None):
        self.successor = successor  # 🔗 Next handler in the chain

    def handle(self, request):
        handled = self.process_request(request)
        if not handled and self.successor:
            self.successor.handle(request)

    def process_request(self, request):
        raise NotImplementedError("Must provide implementation in subclass")

class ConcreteHandler1(Handler):
    def process_request(self, request):
        if 0 < request <= 10:
            print(f"Handler1 processed request: {request}")
            return True

class ConcreteHandler2(Handler):
    def process_request(self, request):
        if 10 < request <= 20:
            print(f"Handler2 processed request: {request}")
            return True

handler = ConcreteHandler1(ConcreteHandler2())
handler.handle(15)  # Outputs: Handler2 processed request: 15
handler.handle(5)   # Outputs: Handler1 processed request: 5

# Example 21: Contextual Behavior with State Pattern
class TrafficLight:
    # 🚦 Traffic light with state pattern
    def __init__(self):
        self.state = RedState()

    def change(self):
        self.state.change(self)

class TrafficLightState(ABC):
    @abstractmethod
    def change(self, light):
        pass

class RedState(TrafficLightState):
    def change(self, light):
        print("Red -> Green")
        light.state = GreenState()

class GreenState(TrafficLightState):
    def change(self, light):
        print("Green -> Yellow")
        light.state = YellowState()

class YellowState(TrafficLightState):
    def change(self, light):
        print("Yellow -> Red")
        light.state = RedState()

light = TrafficLight()
light.change()  # Outputs: Red -> Green
light.change()  # Outputs: Green -> Yellow
light.change()  # Outputs: Yellow -> Red

# Example 22: Internal Class or Nested Class
class Outer:
    # 🌐 Outer class
    class Inner:
        # 🔒 Inner class
        def inner_method(self):
            print("Inner method called")

    def outer_method(self):
        inner = self.Inner()
        inner.inner_method()

outer = Outer()
outer.outer_method()  # Outputs: Inner method called

# Example 23: Custom Iterators with __iter__ and __next__
class Countdown:
    # ⏳ Iterator class for countdown
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self  # Iterator returns itself

    def __next__(self):
        if self.current <= 0:
            raise StopIteration  # 🚫 No more items
        else:
            self.current -= 1
            return self.current + 1

for number in Countdown(5):
    print(number)  # Outputs: 5, 4, 3, 2, 1

# Example 24: Using __getattr__ and __setattr__ for Attribute Access
class DynamicAttributes:
    # 🎩 Class managing attributes dynamically
    def __init__(self):
        self.attributes = {}

    def __getattr__(self, name):
        # Called when attribute not found normally
        return self.attributes.get(name, 'Attribute not found')

    def __setattr__(self, name, value):
        if name == 'attributes':
            super().__setattr__(name, value)
        else:
            self.attributes[name] = value

obj = DynamicAttributes()
obj.new_attr = 'Value'
print(obj.new_attr)        # Outputs: Value
print(obj.nonexistent)     # Outputs: Attribute not found

# Example 25: Applying Mixin Classes for Reusable Features
class LoggerMixin:
    # 📝 Mixin class adding logging functionality
    def log(self, message):
        print(f"[LOG]: {message}")

class Connection:
    # 🌐 Base class for connections
    def connect(self):
        print("Connecting...")

class LoggedConnection(LoggerMixin, Connection):
    # 🔗 Class with logging and connection
    pass

conn = LoggedConnection()
conn.connect()           # Outputs: Connecting...
conn.log("Connection established")  # Outputs: [LOG]: Connection established