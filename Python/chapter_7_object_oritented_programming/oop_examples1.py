# Chapter 7: Object-Oriented Programming (OOP): Modeling the World 🎭
# Building with Blueprints

# 7.1 Introduction to OOP: Thinking in Objects 🎭 (Blueprint Thinking)

# OOP allows us to model real-world entities as objects in code.
# Objects have attributes (data) and methods (behavior).

# Example 1: Defining a simple class
class Vehicle:
    pass  # 'pass' is used as a placeholder for an empty class

# 'Vehicle' is a class that represents the blueprint of a vehicle 🚗

# Example 2: Creating an instance (object) of the class
my_vehicle = Vehicle()
# 'my_vehicle' is an object (instance) of the 'Vehicle' class

# Example 3: Adding attributes to an object dynamically
my_vehicle.type = 'Car'
my_vehicle.brand = 'Toyota'
my_vehicle.color = 'Red'
# Attributes 'type', 'brand', 'color' are added to 'my_vehicle'

# Example 4: Accessing object attributes
print(my_vehicle.type)   # Output: Car
print(my_vehicle.brand)  # Output: Toyota
print(my_vehicle.color)  # Output: Red

# Example 5: Defining a class with an initializer (constructor)
class Vehicle:
    def __init__(self, vehicle_type, brand, color):
        self.type = vehicle_type  # Instance attribute
        self.brand = brand        # Instance attribute
        self.color = color        # Instance attribute

# Example 6: Creating instances with constructor
vehicle1 = Vehicle('Car', 'Honda', 'Blue')
vehicle2 = Vehicle('Motorcycle', 'Yamaha', 'Black')

# Example 7: Accessing instance attributes
print(vehicle1.type)   # Output: Car
print(vehicle2.brand)  # Output: Yamaha

# Example 8: Defining methods within a class
class Vehicle:
    def __init__(self, vehicle_type, brand, color):
        self.type = vehicle_type
        self.brand = brand
        self.color = color

    def start(self):
        print(f"The {self.color} {self.brand} {self.type} is starting. 🚀")

    def stop(self):
        print(f"The {self.color} {self.brand} {self.type} has stopped. 🛑")

# Example 9: Calling methods on an object
vehicle1 = Vehicle('Car', 'Tesla', 'White')
vehicle1.start()  # Output: The White Tesla Car is starting. 🚀
vehicle1.stop()   # Output: The White Tesla Car has stopped. 🛑

# Example 10: Understanding 'self' parameter
# 'self' refers to the instance calling the method, allowing access to its attributes and other methods.

# Example 11: Class attributes vs Instance attributes
class Gadget:
    category = 'Electronics'  # Class attribute shared by all instances

    def __init__(self, name):
        self.name = name  # Instance attribute unique to each instance

# Example 12: Accessing class attributes
gadget1 = Gadget('Smartphone')
gadget2 = Gadget('Laptop')

print(gadget1.category)  # Output: Electronics
print(gadget2.category)  # Output: Electronics

# Example 13: Accessing instance attributes
print(gadget1.name)  # Output: Smartphone
print(gadget2.name)  # Output: Laptop

# Example 14: Modifying class attributes
Gadget.category = 'Tech Devices'  # Changes for all instances
print(gadget1.category)  # Output: Tech Devices

# Example 15: Modifying instance attributes
gadget1.name = 'Tablet'
print(gadget1.name)  # Output: Tablet
print(gadget2.name)  # Output: Laptop

# Example 16: Common mistake - forgetting 'self' in method definitions
class Person:
    def greet():
        print("Hello!")

# Trying to call the method will raise an error
try:
    person = Person()
    person.greet()
except TypeError as e:
    print("Error:", e)
    # Output: Error: greet() takes 0 positional arguments but 1 was given

# Corrected version with 'self' parameter
class Person:
    def greet(self):
        print("Hello! 😊")

person = Person()
person.greet()  # Output: Hello! 😊

# Example 17: Common mistake - assigning to class attributes incorrectly
class Employee:
    count = 0  # Class attribute

    def __init__(self, name):
        self.name = name
        Employee.count += 1  # Correct way to increment class attribute

emp1 = Employee('Alice')
emp2 = Employee('Bob')
print(Employee.count)  # Output: 2

# Example 18: Demonstrating encapsulation
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private attribute (name mangling with __)

    def deposit(self, amount):
        self.__balance += amount
        print(f"Deposited ${amount}. New balance: ${self.__balance}")

    def withdraw(self, amount):
        if amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.__balance}")
        else:
            print("Insufficient funds!")

account = BankAccount(100)
account.deposit(50)     # Deposited $50. New balance: $150
account.withdraw(70)    # Withdrew $70. New balance: $80

# Trying to access private attribute (will raise AttributeError)
try:
    print(account.__balance)
except AttributeError as e:
    print("Error:", e)
    # Output: Error: 'BankAccount' object has no attribute '__balance'

# Correct way to access (not recommended)
print(account._BankAccount__balance)  # Output: 80

# Example 19: Property decorators for getters and setters
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius  # Protected attribute

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero!")
        self._celsius = value

temp = Temperature(25)
print(temp.celsius)  # Output: 25
temp.celsius = 30
print(temp.celsius)  # Output: 30

# Handling exceptions in setter
try:
    temp.celsius = -300
except ValueError as e:
    print("Error:", e)
    # Output: Error: Temperature cannot be below absolute zero!

# Example 20: Defining a class with __str__ and __repr__ methods
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
   
    def __str__(self):
        return f"'{self.title}' by {self.author}"  # For users

    def __repr__(self):
        return f"Book(title='{self.title}', author='{self.author}')"  # For developers

book = Book('1984', 'George Orwell')
print(str(book))   # Output: '1984' by George Orwell
print(repr(book))  # Output: Book(title='1984', author='George Orwell')

# Example 21: Class methods and static methods
class MathOperations:
    @staticmethod
    def add(a, b):
        return a + b

    @classmethod
    def info(cls):
        print(f"This is {cls.__name__} class.")

print(MathOperations.add(5, 7))  # Output: 12
MathOperations.info()            # Output: This is MathOperations class.

# Example 22: Common mistake - improper use of class methods
class Incorrect:
    def method():
        print("Method without parameters")

# Calling method will cause an error
try:
    obj = Incorrect()
    obj.method()
except TypeError as e:
    print("Error:", e)
    # Output: Error: method() takes 0 positional arguments but 1 was given

# Correct way with 'self'
class Correct:
    def method(self):
        print("Method with self parameter")

obj = Correct()
obj.method()  # Output: Method with self parameter

# 7.2 Classes and Objects: Blueprints and Instances Blueprint ➡️ 🚗

# Example 1: Defining a class 'Animal' with attributes and methods
class Animal:
    def __init__(self, name, species):
        self.name = name       # Instance attribute
        self.species = species # Instance attribute

    def make_sound(self):
        print(f"{self.name} makes a sound.")

# Example 2: Creating instances of 'Animal'
animal1 = Animal('Lion', 'Felidae')
animal2 = Animal('Wolf', 'Canidae')

# Example 3: Calling methods on instances
animal1.make_sound()  # Output: Lion makes a sound.
animal2.make_sound()  # Output: Wolf makes a sound.

# Example 4: Defining subclasses (will cover more in inheritance)

# Example 5: Understanding 'self' is not a keyword, can be named differently (but not recommended)
class Sample:
    def __init__(xyz, value):  # 'xyz' instead of 'self' (not recommended)
        xyz.value = value

    def display(xyz):
        print(f"Value is {xyz.value}")

sample = Sample(10)
sample.display()  # Output: Value is 10

# Example 6: Creating multiple instances with different states
class Counter:
    def __init__(self):
        self.count = 0  # Each instance has its own 'count'

    def increment(self):
        self.count += 1

counter1 = Counter()
counter2 = Counter()

counter1.increment()
counter1.increment()
counter2.increment()

print(counter1.count)  # Output: 2
print(counter2.count)  # Output: 1

# Example 7: Using __dict__ to see attributes
print(counter1.__dict__)  # Output: {'count': 2}

# Example 8: Modifying class attributes
class GameSettings:
    difficulty = 'Easy'  # Class attribute

print(GameSettings.difficulty)  # Output: Easy

# Changing class attribute
GameSettings.difficulty = 'Hard'
print(GameSettings.difficulty)  # Output: Hard

# Example 9: Instance vs Class attributes with same name
class SampleClass:
    var = 'Class variable'

obj = SampleClass()
obj.var = 'Instance variable'

print(obj.var)             # Output: Instance variable
print(SampleClass.var)     # Output: Class variable

# Example 10: Deleting instance attribute
del obj.var
print(obj.var)  # Output: Class variable (falls back to class attribute)

# Example 11: Using hasattr(), getattr(), setattr(), delattr()
class Person:
    def __init__(self, name):
        self.name = name

person = Person('Dave')

# Check if attribute exists
print(hasattr(person, 'name'))  # Output: True

# Get attribute value
print(getattr(person, 'name'))  # Output: Dave

# Set attribute value
setattr(person, 'age', 30)
print(person.age)  # Output: 30

# Delete attribute
delattr(person, 'age')
print(hasattr(person, 'age'))  # Output: False

# Example 12: Private and protected attributes
class Test:
    def __init__(self):
        self._protected = 'Protected'
        self.__private = 'Private'

test = Test()
print(test._protected)   # Output: Protected
# Accessing private attribute (will raise AttributeError)
try:
    print(test.__private)
except AttributeError as e:
    print("Error:", e)
    # Output: Error: 'Test' object has no attribute '__private'

# Accessing name-mangled private attribute
print(test._Test__private)  # Output: Private

# Example 13: Static variables in classes
class Example:
    static_var = 5  # Class variable

    def __init__(self):
        self.instance_var = 10  # Instance variable

print(Example.static_var)  # Output: 5

example = Example()
print(example.static_var)    # Output: 5
print(example.instance_var)  # Output: 10

# Example 14: Calling methods from methods
class Calculator:
    def __init__(self, value):
        self.value = value

    def add(self, amount):
        self.value += amount
        return self  # Returning self for method chaining

    def subtract(self, amount):
        self.value -= amount
        return self

calc = Calculator(10)
calc.add(5).subtract(3)  # Method chaining
print(calc.value)        # Output: 12

# Example 15: Using properties to control attribute access
class Rectangle:
    def __init__(self, width, height):
        self.width = width    # Calls setter
        self.height = height  # Calls setter

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("Width must be positive!")
        self._width = value

    @property
    def area(self):
        return self._width * self.height

rect = Rectangle(5, 10)
print(rect.area)  # Output: 50

# Handling exceptions in setter
try:
    rect.width = -3
except ValueError as e:
    print("Error:", e)
    # Output: Error: Width must be positive!

# 7.3 Inheritance: Building Upon Existing Classes 🧬 (Family Tree)

# Example 1: Defining a parent class 'Animal'
class Animal:
    def __init__(self, name):
        self.name = name

    def move(self):
        print(f"{self.name} moves.")

# Example 2: Defining a child class 'Dog' that inherits from 'Animal'
class Dog(Animal):
    def bark(self):
        print(f"{self.name} says Woof! 🐶")

# Example 3: Creating an instance of 'Dog'
dog = Dog('Buddy')
dog.move()  # Inherited method: Buddy moves.
dog.bark()  # Specific method: Buddy says Woof! 🐶

# Example 4: Method overriding
class Bird(Animal):
    def move(self):
        print(f"{self.name} flies. 🐦")

bird = Bird('Tweety')
bird.move()  # Output: Tweety flies. 🐦

# Example 5: Using super() to call parent class methods
class Horse(Animal):
    def move(self):
        super().move()  # Calls the move method of 'Animal'
        print(f"{self.name} gallops. 🐎")

horse = Horse('Spirit')
horse.move()
# Output:
# Spirit moves.
# Spirit gallops. 🐎

# Example 6: Multiple inheritance
class Flyer:
    def fly(self):
        print("This object can fly. ✈️")

class Swimmer:
    def swim(self):
        print("This object can swim. 🏊‍♂️")

class FlyingFish(Flyer, Swimmer):
    pass

fish = FlyingFish()
fish.fly()   # Output: This object can fly. ✈️
fish.swim()  # Output: This object can swim. 🏊‍♂️

# Example 7: Overriding __init__ in child class
class Employee:
    def __init__(self, name):
        self.name = name

class Manager(Employee):
    def __init__(self, name, department):
        super().__init__(name)
        self.department = department

manager = Manager('Alice', 'HR')
print(manager.name)        # Output: Alice
print(manager.department)  # Output: HR

# Example 8: Checking isinstance() and issubclass()
print(isinstance(manager, Manager))    # Output: True
print(isinstance(manager, Employee))   # Output: True
print(issubclass(Manager, Employee))   # Output: True
print(issubclass(Employee, Manager))   # Output: False

# Example 9: Method Resolution Order (MRO)
class A:
    def do_something(self):
        print("Method from class A")

class B(A):
    def do_something(self):
        print("Method from class B")

class C(A):
    def do_something(self):
        print("Method from class C")

class D(B, C):
    pass

d = D()
d.do_something()
# Output: Method from class B (B comes before C in inheritance)

# Example 10: Accessing parent methods with super()
class Base:
    def greet(self):
        print("Hello from Base class")

class Derived(Base):
    def greet(self):
        super().greet()
        print("Hello from Derived class")

obj = Derived()
obj.greet()
# Output:
# Hello from Base class
# Hello from Derived class

# Example 11: Abstract base classes (using abc module)
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

# Cannot instantiate abstract class
try:
    shape = Shape()
except TypeError as e:
    print("Error:", e)
    # Output: Error: Can't instantiate abstract class Shape with abstract methods area

# Example 12: Implementing abstract methods
class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

square = Square(5)
print(square.area())  # Output: 25

# Example 13: Protected members and inheritance
class Parent:
    def __init__(self):
        self._protected = 'Protected'

class Child(Parent):
    def access_protected(self):
        print(self._protected)

child = Child()
child.access_protected()  # Output: Protected

# Example 14: Using super() with multiple inheritance (MRO)
class X:
    def do_something(self):
        print("Method from class X")

class Y(X):
    def do_something(self):
        print("Method from class Y")
        super().do_something()

class Z(X):
    def do_something(self):
        print("Method from class Z")
        super().do_something()

class M(Y, Z):
    def do_something(self):
        print("Method from class M")
        super().do_something()

m = M()
m.do_something()
# Output:
# Method from class M
# Method from class Y
# Method from class Z
# Method from class X

# Example 15: Overriding class methods
class MyClass:
    @classmethod
    def greet(cls):
        print(f"Hello from {cls.__name__}")

class MySubClass(MyClass):
    @classmethod
    def greet(cls):
        print(f"Greetings from {cls.__name__}")

MyClass.greet()        # Output: Hello from MyClass
MySubClass.greet()     # Output: Greetings from MySubClass

# Example 16: Common mistake - not calling super().__init__()
class BaseClass:
    def __init__(self):
        self.base_attribute = 'Base'

class DerivedClass(BaseClass):
    def __init__(self):
        self.derived_attribute = 'Derived'

# Missing super().__init__() means base attributes are not initialized
obj = DerivedClass()
try:
    print(obj.base_attribute)
except AttributeError as e:
    print("Error:", e)
    # Output: Error: 'DerivedClass' object has no attribute 'base_attribute'

# Corrected version
class DerivedClass(BaseClass):
    def __init__(self):
        super().__init__()
        self.derived_attribute = 'Derived'

obj = DerivedClass()
print(obj.base_attribute)  # Output: Base

# 7.4 Polymorphism: Many Forms, One Interface 🎭 (Adaptable Actions)

# Example 1: Different classes with the same method name
class Cat:
    def speak(self):
        print("Meow! 🐱")

class Dog:
    def speak(self):
        print("Woof! 🐶")

# Example 2: Function using polymorphism
def animal_speak(animal):
    animal.speak()  # Method called depends on the object's class

cat = Cat()
dog = Dog()

animal_speak(cat)  # Output: Meow! 🐱
animal_speak(dog)  # Output: Woof! 🐶

# Example 3: Polymorphism with inheritance
class Shape:
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

shapes = [Rectangle(3, 4), Circle(5)]
for shape in shapes:
    print(shape.area())  # Output depends on the shape

# Output:
# 12
# 78.5

# Example 4: Duck typing (focus on methods rather than class type)
class Whale:
    def swim(self):
        print("Whale swims. 🐋")

class Submarine:
    def swim(self):
        print("Submarine moves underwater. 🚢")

def make_it_swim(swimmer):
    swimmer.swim()

whale = Whale()
submarine = Submarine()

make_it_swim(whale)      # Output: Whale swims. 🐋
make_it_swim(submarine)  # Output: Submarine moves underwater. 🚢

# Example 5: Polymorphism with built-in functions
print(len("Hello"))      # Output: 5
print(len([1, 2, 3, 4])) # Output: 4
print(len({'a': 1, 'b': 2}))  # Output: 2

# Example 6: Operator overloading
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Overloading the '+' operator
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(2, 3)
v2 = Vector(5, 7)
v3 = v1 + v2  # Uses __add__ method
print(f"v3 = ({v3.x}, {v3.y})")  # Output: v3 = (7, 10)

# Example 7: Implementing the same interface
class JSONSerializer:
    def serialize(self, obj):
        import json
        return json.dumps(obj)

class XMLSerializer:
    def serialize(self, obj):
        # Simplified example
        return f"<data>{str(obj)}</data>"

def serialize_object(serializer, obj):
    print(serializer.serialize(obj))

data = {'name': 'Alice', 'age': 30}

json_serializer = JSONSerializer()
xml_serializer = XMLSerializer()

serialize_object(json_serializer, data)
# Output: {"name": "Alice", "age": 30}

serialize_object(xml_serializer, data)
# Output: <data>{'name': 'Alice', 'age': 30}</data>

# Example 8: Polymorphism with custom iterable objects
class ReverseIterable:
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -=1
        return self.data[self.index]

rev = ReverseIterable([1, 2, 3, 4])
for item in rev:
    print(item)
# Output:
# 4
# 3
# 2
# 1

# Example 9: Common interface for file-like objects
class FileA:
    def read(self):
        return "Data from FileA"

class FileB:
    def read(self):
        return "Data from FileB"

def read_data(file_obj):
    data = file_obj.read()
    print(data)

file_a = FileA()
file_b = FileB()

read_data(file_a)  # Output: Data from FileA
read_data(file_b)  # Output: Data from FileB

# Example 10: Method overriding and polymorphism
class Employee:
    def work(self):
        print("Employee works.")

class Programmer(Employee):
    def work(self):
        print("Programmer writes code.")

class Designer(Employee):
    def work(self):
        print("Designer creates designs.")

employees = [Employee(), Programmer(), Designer()]
for emp in employees:
    emp.work()
# Output:
# Employee works.
# Programmer writes code.
# Designer creates designs.

# Example 11: Polymorphism with abstract base class
from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self, message):
        pass

class EmailNotification(Notification):
    def send(self, message):
        print(f"Sending email: {message}")

class SMSNotification(Notification):
    def send(self, message):
        print(f"Sending SMS: {message}")

def notify(notification, message):
    notification.send(message)

email_notif = EmailNotification()
sms_notif = SMSNotification()

notify(email_notif, "Hello via Email!")
notify(sms_notif, "Hello via SMS!")
# Output:
# Sending email: Hello via Email!
# Sending SMS: Hello via SMS!

# Example 12: Polymorphic behavior with built-in types
data_list = [1, 2, 3]
data_tuple = (4, 5, 6)

print(len(data_list))   # Output: 3
print(len(data_tuple))  # Output: 3

# Example 13: Custom objects supporting len()
class CustomCollection:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

collection = CustomCollection(['apple', 'banana', 'cherry'])
print(len(collection))  # Output: 3

# Example 14: Error handling in polymorphic methods
class Animal:
    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

class Dog(Animal):
    def speak(self):
        print("Woof!")

class Fish(Animal):
    pass  # Does not implement 'speak'

dog = Dog()
dog.speak()  # Output: Woof!

fish = Fish()
try:
    fish.speak()
except NotImplementedError as e:
    print("Error:", e)
    # Output: Error: Subclasses must implement this method

# Example 15: Polymorphism with function arguments
def add(a, b):
    return a + b

print(add(5, 3))            # Output: 8 (integers)
print(add("Hello, ", "World"))  # Output: Hello, World (strings)
print(add([1, 2], [3, 4]))  # Output: [1, 2, 3, 4] (lists)

# This concludes the detailed examples for Object-Oriented Programming in Python.
# We've covered classes, objects, encapsulation, inheritance, and polymorphism.
# Each example is designed to provide a deep understanding of OOP concepts.
# Remember to practice and experiment with these concepts to master OOP! 🧑‍💻