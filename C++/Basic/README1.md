# Python Built-in Functions

## A

### abs()

**Description**: Returns the absolute value of a number.

**Example**:
```python
abs(-5)
# Output: 5
```

### aiter()

**Description**: Returns an asynchronous iterator for an object.

**Example**:
```python
# Async iterator example
async def main():
    async for value in aiter([1, 2, 3]):
        print(value)
# Output: 1
#         2
#         3
```

### all()

**Description**: Returns True if all elements of the iterable are true (or if the iterable is empty).

**Example**:
```python
all([True, True, False])
# Output: False
```

### anext()

**Description**: Retrieves the next item from an asynchronous iterator.

**Example**:
```python
# Async next example
async def main():
    itr = aiter([1, 2, 3])
    print(await anext(itr))
# Output: 1
```

### any()

**Description**: Returns True if any element of the iterable is true. If the iterable is empty, returns False.

**Example**:
```python
any([True, False, False])
# Output: True
```

### ascii()

**Description**: Returns a string containing a printable representation of an object, but escapes non-ASCII characters.

**Example**:
```python
ascii('�')
# Output: '\xf1'
```

## B

### bin()

**Description**: Converts an integer number to a binary string prefixed with '0b'.

**Example**:
```python
bin(5)
# Output: '0b101'
```

### bool()

**Description**: Converts a value to a Boolean, using the standard truth testing procedure.

**Example**:
```python
bool(0)
# Output: False
```

### breakpoint()

**Description**: Drops you into the debugger at the call site.

**Example**:
```python
breakpoint()
# Output: Starts the debugger
```

### bytearray()

**Description**: Returns a new array of bytes.

**Example**:
```python
bytearray('abc', 'utf-8')
# Output: bytearray(b'abc')
```

### bytes()

**Description**: Returns a new 'bytes' object, which is an immutable sequence of bytes.

**Example**:
```python
bytes('abc', 'utf-8')
# Output: b'abc'
```

## C

### callable()

**Description**: Returns True if the object appears callable, False if not.

**Example**:
```python
callable(len)
# Output: True
```

### chr()

**Description**: Returns the string representing a character whose Unicode code is the integer.

**Example**:
```python
chr(97)
# Output: 'a'
```

### classmethod()

**Description**: Returns a class method for a given function.

**Example**:
```python
class MyClass:
    @classmethod
    def my_classmethod(cls):
        pass
# Usage: MyClass.my_classmethod()
```

### compile()

**Description**: Compiles source into a code or AST object.

**Example**:
```python
code = 'print("Hello World")'
compiled_code = compile(code, '<string>', 'exec')
exec(compiled_code)
```

### complex()

**Description**: Creates a complex number.

**Example**:
```python
complex(1, 2)
# Output: (1+2j)
```

## D

### delattr()

**Description**: Deletes the named attribute from an object.

**Example**:
```python
class MyClass:
    pass
obj = MyClass()
obj.attr = 10
delattr(obj, 'attr')
```

### dict()

**Description**: Creates a new dictionary.

**Example**:
```python
dict(a=1, b=2)
# Output: {'a': 1, 'b': 2}
```

### dir()

**Description**: Returns a list of valid attributes for the object.

**Example**:
```python
dir([])
# Output: ['__add__', '__class__', ...]
```

### divmod()

**Description**: Takes two numbers and returns a pair of numbers (a tuple) consisting of their quotient and remainder.

**Example**:
```python
divmod(9, 4)
# Output: (2, 1)
```

## E

### enumerate()

**Description**: Adds a counter to an iterable and returns it.

**Example**:
```python
enumerate(['a', 'b', 'c'])
# Output: [(0, 'a'), (1, 'b'), (2, 'c')]
```

### eval()

**Description**: Parses the expression passed to it and runs python expression.

**Example**:
```python
eval('1 + 2')
# Output: 3
```

### exec()

**Description**: Executes the dynamically created program, which is either a string or a code object.

**Example**:
```python
exec('print("Hello World")')
```

## F

### filter()

**Description**: Constructs an iterator from those elements of iterable for which function returns true.

**Example**:
```python
list(filter(lambda x: x > 0, [-1, 0, 1]))
# Output: [1]
```

### float()

**Description**: Returns a floating-point number.

**Example**:
```python
float('3.14')
# Output: 3.14
```

### format()

**Description**: Formats a specified value.

**Example**:
```python
format(3.14159, '.2f')
# Output: '3.14'
```

### frozenset()

**Description**: Returns an immutable frozenset object.

**Example**:
```python
frozenset([1, 2, 3, 1])
# Output: frozenset({1, 2, 3})
```

## G

### getattr()

**Description**: Returns the value of the named attribute of an object.

**Example**:
```python
class MyClass:
    attr = 1
getattr(MyClass, 'attr')
# Output: 1
```

### globals()

**Description**: Returns a dictionary representing the current global symbol table.

**Example**:
```python
globals()
# Output: {'__name__': '__main__', ...}
```

## H

### hasattr()

**Description**: Returns True if the object has the named attribute.

**Example**:
```python
class MyClass:
    attr = 1
hasattr(MyClass, 'attr')
# Output: True
```

### hash()

**Description**: Returns the hash value of the object (if it has one).

**Example**:
```python
hash('test')
# Output: hash value
```

### help()

**Description**: Invokes the built-in help system.

**Example**:
```python
help(str)
# Output: Help on class str in module builtins
```

### hex()

**Description**: Converts an integer number to a lowercase hexadecimal string prefixed with '0x'.

**Example**:
```python
hex(255)
# Output: '0xff'
```

## I

### id()

**Description**: Returns the identity of an object.

**Example**:
```python
id(object())
# Output: Unique identifier
```

### input()

**Description**: Reads a line from input, converts it to a string (stripping a trailing newline).

**Example**:
```python
input('Enter something: ')
# Output: User input
```

### int()

**Description**: Converts a number or string to an integer.

**Example**:
```python
int('123')
# Output: 123
```

### isinstance()

**Description**: Returns True if the specified object is of the specified type.

**Example**:
```python
isinstance(1, int)
# Output: True
```

### issubclass()

**Description**: Returns True if the specified class is a subclass of the specified class.

**Example**:
```python
issubclass(bool, int)
# Output: True
```

### iter()

**Description**: Returns an iterator object.

**Example**:
```python
iter([1, 2, 3])
# Output: <list_iterator>
```

## L

### len()

**Description**: Returns the length (the number of items) of an object.

**Example**:
```python
len([1, 2, 3])
# Output: 3
```

### list()

**Description**: Returns a list.

**Example**:
```python
list((1, 2, 3))
# Output: [1, 2, 3]
```

### locals()

**Description**: Updates and returns a dictionary representing the current local symbol table.

**Example**:
```python
def test():
    a = 1
    print(locals())
test()
# Output: {'a': 1}
```

## M

### map()

**Description**: Applies a function to every item of an iterable.

**Example**:
```python
list(map(lambda x: x*2, [1, 2, 3]))
# Output: [2, 4, 6]
```

### max()

**Description**: Returns the largest item in an iterable.

**Example**:
```python
max([1, 2, 3])
# Output: 3
```

### memoryview()

**Description**: Returns a memory view object.

**Example**:
```python
memoryview(b'abc')
# Output: <memory at 0x...>
```

### min()

**Description**: Returns the smallest item in an iterable.

**Example**:
```python
min([1, 2, 3])
# Output: 1
```

## N

### next()

**Description**: Retrieves the next item from an iterator.

**Example**:
```python
next(iter([1, 2, 3]))
# Output: 1
```

## O

### object()

**Description**: Returns a new featureless object.

**Example**:
```python
object()
# Output: <object object at 0x...>
```

### oct()

**Description**: Converts an integer number to an octal string prefixed with '0o'.

**Example**:
```python
oct(8)
# Output: '0o10'
```

### open()

**Description**: Opens a file and returns a corresponding file object.

**Example**:
```python
open('file.txt', 'r')
# Output: <_io.TextIOWrapper>
```

### ord()

**Description**: Returns the Unicode code of a given character.

**Example**:
```python
ord('a')
# Output: 97
```

## P

### pow()

**Description**: Returns the value of x to the power of y.

**Example**:
```python
pow(2, 3)
# Output: 8
```

### print()

**Description**: Prints the given object to the console.

**Example**:
```python
print('Hello, World!')
# Output: Hello, World!
```

### property()

**Description**: Returns a property attribute.

**Example**:
```python
class C:
    def __init__(self):
        self._x = None
    def getx(self):
        return self._x
    def setx(self, value):
        self._x = value
    x = property(getx, setx)
# Usage: c = C()
#        c.x = 1
#        print(c.x)
```

## R

### range()

**Description**: Returns a sequence of numbers.

**Example**:
```python
range(5)
# Output: range(0, 5)
```

### repr()

**Description**: Returns a string containing a printable representation of an object.

**Example**:
```python
repr('test')
# Output: "'test'"
```

### reversed()

**Description**: Returns a reversed iterator.

**Example**:
```python
list(reversed([1, 2, 3]))
# Output: [3, 2, 1]
```

### round()

**Description**: Rounds a number to a given precision in decimal digits.

**Example**:
```python
round(3.14159, 2)
# Output: 3.14
```

## S

### set()

**Description**: Returns a new set object.

**Example**:
```python
set([1, 2, 3, 1])
# Output: {1, 2, 3}
```

### setattr()

**Description**: Sets the value of the named attribute of an object.

**Example**:
```python
class MyClass:
    pass
obj = MyClass()
setattr(obj, 'attr', 10)
print(obj.attr)
# Output: 10
```

### slice()

**Description**: Returns a slice object.

**Example**:
```python
slice(1, 5, 2)
# Output: slice(1, 5, 2)
```

### sorted()

**Description**: Returns a new sorted list from the items in an iterable.

**Example**:
```python
sorted([3, 1, 2])
# Output: [1, 2, 3]
```

### staticmethod()

**Description**: Returns a static method for a given function.

**Example**:
```python
class MyClass:
    @staticmethod
    def my_staticmethod():
        pass
# Usage: MyClass.my_staticmethod()
```

### str()

**Description**: Returns a string version of an object.

**Example**:
```python
str(123)
# Output: '123'
```

### sum()

**Description**: Returns the sum of a 'start' value (default: 0) plus an iterable of numbers.

**Example**:
```python
sum([1, 2, 3])
# Output: 6
```

### super()

**Description**: Returns a proxy object that delegates method calls to a parent or sibling class.

**Example**:
```python
class Base:
    def __init__(self):
        print('Base')
class Derived(Base):
    def __init__(self):
        super().__init__()
        print('Derived')
# Usage: Derived()
# Output: Base
#         Derived
```

## T

### tuple()

**Description**: Returns a tuple.

**Example**:
```python
tuple([1, 2, 3])
# Output: (1, 2, 3)
```

### type()

**Description**: Returns the type of an object.

**Example**:
```python
type(123)
# Output: <class 'int'>
```

## V

### vars()

**Description**: Returns the __dict__ attribute of the given object.

**Example**:
```python
class MyClass:
    attr = 1
obj = MyClass()
vars(obj)
# Output: {'attr': 1}
```

## Z

### zip()

**Description**: Returns an iterator of tuples, where the i-th tuple contains the i-th element from each of the argument sequences.

**Example**:
```python
list(zip([1, 2, 3], ['a', 'b', 'c']))
# Output: [(1, 'a'), (2, 'b'), (3, 'c')]
```

## _

### __import__()

**Description**: Imports a module.

**Example**:
```python
__import__('math')
# Output: <module 'math'>
```

