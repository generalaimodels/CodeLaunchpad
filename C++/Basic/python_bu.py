def generate_readme():
    built_in_functions = {
        "A": [
            {"name": "abs()", "example": "abs(-5)\n# Output: 5", "description": "Returns the absolute value of a number."},
            {"name": "aiter()", "example": "# Async iterator example\nasync def main():\n    async for value in aiter([1, 2, 3]):\n        print(value)\n# Output: 1\n#         2\n#         3", "description": "Returns an asynchronous iterator for an object."},
            {"name": "all()", "example": "all([True, True, False])\n# Output: False", "description": "Returns True if all elements of the iterable are true (or if the iterable is empty)."},
            {"name": "anext()", "example": "# Async next example\nasync def main():\n    itr = aiter([1, 2, 3])\n    print(await anext(itr))\n# Output: 1", "description": "Retrieves the next item from an asynchronous iterator."},
            {"name": "any()", "example": "any([True, False, False])\n# Output: True", "description": "Returns True if any element of the iterable is true. If the iterable is empty, returns False."},
            {"name": "ascii()", "example": "ascii('ñ')\n# Output: '\\xf1'", "description": "Returns a string containing a printable representation of an object, but escapes non-ASCII characters."}
        ],
        "B": [
            {"name": "bin()", "example": "bin(5)\n# Output: '0b101'", "description": "Converts an integer number to a binary string prefixed with '0b'."},
            {"name": "bool()", "example": "bool(0)\n# Output: False", "description": "Converts a value to a Boolean, using the standard truth testing procedure."},
            {"name": "breakpoint()", "example": "breakpoint()\n# Output: Starts the debugger", "description": "Drops you into the debugger at the call site."},
            {"name": "bytearray()", "example": "bytearray('abc', 'utf-8')\n# Output: bytearray(b'abc')", "description": "Returns a new array of bytes."},
            {"name": "bytes()", "example": "bytes('abc', 'utf-8')\n# Output: b'abc'", "description": "Returns a new 'bytes' object, which is an immutable sequence of bytes."}
        ],
        "C": [
            {"name": "callable()", "example": "callable(len)\n# Output: True", "description": "Returns True if the object appears callable, False if not."},
            {"name": "chr()", "example": "chr(97)\n# Output: 'a'", "description": "Returns the string representing a character whose Unicode code is the integer."},
            {"name": "classmethod()", "example": "class MyClass:\n    @classmethod\n    def my_classmethod(cls):\n        pass\n# Usage: MyClass.my_classmethod()", "description": "Returns a class method for a given function."},
            {"name": "compile()", "example": "code = 'print(\"Hello World\")'\ncompiled_code = compile(code, '<string>', 'exec')\nexec(compiled_code)", "description": "Compiles source into a code or AST object."},
            {"name": "complex()", "example": "complex(1, 2)\n# Output: (1+2j)", "description": "Creates a complex number."}
        ],
        "D": [
            {"name": "delattr()", "example": "class MyClass:\n    pass\nobj = MyClass()\nobj.attr = 10\ndelattr(obj, 'attr')", "description": "Deletes the named attribute from an object."},
            {"name": "dict()", "example": "dict(a=1, b=2)\n# Output: {'a': 1, 'b': 2}", "description": "Creates a new dictionary."},
            {"name": "dir()", "example": "dir([])\n# Output: ['__add__', '__class__', ...]", "description": "Returns a list of valid attributes for the object."},
            {"name": "divmod()", "example": "divmod(9, 4)\n# Output: (2, 1)", "description": "Takes two numbers and returns a pair of numbers (a tuple) consisting of their quotient and remainder."}
        ],
        "E": [
            {"name": "enumerate()", "example": "enumerate(['a', 'b', 'c'])\n# Output: [(0, 'a'), (1, 'b'), (2, 'c')]", "description": "Adds a counter to an iterable and returns it."},
            {"name": "eval()", "example": "eval('1 + 2')\n# Output: 3", "description": "Parses the expression passed to it and runs python expression."},
            {"name": "exec()", "example": "exec('print(\"Hello World\")')", "description": "Executes the dynamically created program, which is either a string or a code object."}
        ],
        "F": [
            {"name": "filter()", "example": "list(filter(lambda x: x > 0, [-1, 0, 1]))\n# Output: [1]", "description": "Constructs an iterator from those elements of iterable for which function returns true."},
            {"name": "float()", "example": "float('3.14')\n# Output: 3.14", "description": "Returns a floating-point number."},
            {"name": "format()", "example": "format(3.14159, '.2f')\n# Output: '3.14'", "description": "Formats a specified value."},
            {"name": "frozenset()", "example": "frozenset([1, 2, 3, 1])\n# Output: frozenset({1, 2, 3})", "description": "Returns an immutable frozenset object."}
        ],
        "G": [
            {"name": "getattr()", "example": "class MyClass:\n    attr = 1\ngetattr(MyClass, 'attr')\n# Output: 1", "description": "Returns the value of the named attribute of an object."},
            {"name": "globals()", "example": "globals()\n# Output: {'__name__': '__main__', ...}", "description": "Returns a dictionary representing the current global symbol table."}
        ],
        "H": [
            {"name": "hasattr()", "example": "class MyClass:\n    attr = 1\nhasattr(MyClass, 'attr')\n# Output: True", "description": "Returns True if the object has the named attribute."},
            {"name": "hash()", "example": "hash('test')\n# Output: hash value", "description": "Returns the hash value of the object (if it has one)."},
            {"name": "help()", "example": "help(str)\n# Output: Help on class str in module builtins", "description": "Invokes the built-in help system."},
            {"name": "hex()", "example": "hex(255)\n# Output: '0xff'", "description": "Converts an integer number to a lowercase hexadecimal string prefixed with '0x'."}
        ],
        "I": [
            {"name": "id()", "example": "id(object())\n# Output: Unique identifier", "description": "Returns the identity of an object."},
            {"name": "input()", "example": "input('Enter something: ')\n# Output: User input", "description": "Reads a line from input, converts it to a string (stripping a trailing newline)."},
            {"name": "int()", "example": "int('123')\n# Output: 123", "description": "Converts a number or string to an integer."},
            {"name": "isinstance()", "example": "isinstance(1, int)\n# Output: True", "description": "Returns True if the specified object is of the specified type."},
            {"name": "issubclass()", "example": "issubclass(bool, int)\n# Output: True", "description": "Returns True if the specified class is a subclass of the specified class."},
            {"name": "iter()", "example": "iter([1, 2, 3])\n# Output: <list_iterator>", "description": "Returns an iterator object."}
        ],
        "L": [
            {"name": "len()", "example": "len([1, 2, 3])\n# Output: 3", "description": "Returns the length (the number of items) of an object."},
            {"name": "list()", "example": "list((1, 2, 3))\n# Output: [1, 2, 3]", "description": "Returns a list."},
            {"name": "locals()", "example": "def test():\n    a = 1\n    print(locals())\ntest()\n# Output: {'a': 1}", "description": "Updates and returns a dictionary representing the current local symbol table."}
        ],
        "M": [
            {"name": "map()", "example": "list(map(lambda x: x*2, [1, 2, 3]))\n# Output: [2, 4, 6]", "description": "Applies a function to every item of an iterable."},
            {"name": "max()", "example": "max([1, 2, 3])\n# Output: 3", "description": "Returns the largest item in an iterable."},
            {"name": "memoryview()", "example": "memoryview(b'abc')\n# Output: <memory at 0x...>", "description": "Returns a memory view object."},
            {"name": "min()", "example": "min([1, 2, 3])\n# Output: 1", "description": "Returns the smallest item in an iterable."}
        ],
        "N": [
            {"name": "next()", "example": "next(iter([1, 2, 3]))\n# Output: 1", "description": "Retrieves the next item from an iterator."}
        ],
        "O": [
            {"name": "object()", "example": "object()\n# Output: <object object at 0x...>", "description": "Returns a new featureless object."},
            {"name": "oct()", "example": "oct(8)\n# Output: '0o10'", "description": "Converts an integer number to an octal string prefixed with '0o'."},
            {"name": "open()", "example": "open('file.txt', 'r')\n# Output: <_io.TextIOWrapper>", "description": "Opens a file and returns a corresponding file object."},
            {"name": "ord()", "example": "ord('a')\n# Output: 97", "description": "Returns the Unicode code of a given character."}
        ],
        "P": [
            {"name": "pow()", "example": "pow(2, 3)\n# Output: 8", "description": "Returns the value of x to the power of y."},
            {"name": "print()", "example": "print('Hello, World!')\n# Output: Hello, World!", "description": "Prints the given object to the console."},
            {"name": "property()", "example": "class C:\n    def __init__(self):\n        self._x = None\n    def getx(self):\n        return self._x\n    def setx(self, value):\n        self._x = value\n    x = property(getx, setx)\n# Usage: c = C()\n#        c.x = 1\n#        print(c.x)", "description": "Returns a property attribute."}
        ],
        "R": [
            {"name": "range()", "example": "range(5)\n# Output: range(0, 5)", "description": "Returns a sequence of numbers."},
            {"name": "repr()", "example": "repr('test')\n# Output: \"'test'\"", "description": "Returns a string containing a printable representation of an object."},
            {"name": "reversed()", "example": "list(reversed([1, 2, 3]))\n# Output: [3, 2, 1]", "description": "Returns a reversed iterator."},
            {"name": "round()", "example": "round(3.14159, 2)\n# Output: 3.14", "description": "Rounds a number to a given precision in decimal digits."}
        ],
        "S": [
            {"name": "set()", "example": "set([1, 2, 3, 1])\n# Output: {1, 2, 3}", "description": "Returns a new set object."},
            {"name": "setattr()", "example": "class MyClass:\n    pass\nobj = MyClass()\nsetattr(obj, 'attr', 10)\nprint(obj.attr)\n# Output: 10", "description": "Sets the value of the named attribute of an object."},
            {"name": "slice()", "example": "slice(1, 5, 2)\n# Output: slice(1, 5, 2)", "description": "Returns a slice object."},
            {"name": "sorted()", "example": "sorted([3, 1, 2])\n# Output: [1, 2, 3]", "description": "Returns a new sorted list from the items in an iterable."},
            {"name": "staticmethod()", "example": "class MyClass:\n    @staticmethod\n    def my_staticmethod():\n        pass\n# Usage: MyClass.my_staticmethod()", "description": "Returns a static method for a given function."},
            {"name": "str()", "example": "str(123)\n# Output: '123'", "description": "Returns a string version of an object."},
            {"name": "sum()", "example": "sum([1, 2, 3])\n# Output: 6", "description": "Returns the sum of a 'start' value (default: 0) plus an iterable of numbers."},
            {"name": "super()", "example": "class Base:\n    def __init__(self):\n        print('Base')\nclass Derived(Base):\n    def __init__(self):\n        super().__init__()\n        print('Derived')\n# Usage: Derived()\n# Output: Base\n#         Derived", "description": "Returns a proxy object that delegates method calls to a parent or sibling class."}
        ],
        "T": [
            {"name": "tuple()", "example": "tuple([1, 2, 3])\n# Output: (1, 2, 3)", "description": "Returns a tuple."},
            {"name": "type()", "example": "type(123)\n# Output: <class 'int'>", "description": "Returns the type of an object."}
        ],
        "V": [
            {"name": "vars()", "example": "class MyClass:\n    attr = 1\nobj = MyClass()\nvars(obj)\n# Output: {'attr': 1}", "description": "Returns the __dict__ attribute of the given object."}
        ],
        "Z": [
            {"name": "zip()", "example": "list(zip([1, 2, 3], ['a', 'b', 'c']))\n# Output: [(1, 'a'), (2, 'b'), (3, 'c')]", "description": "Returns an iterator of tuples, where the i-th tuple contains the i-th element from each of the argument sequences."}
        ],
        "_": [
            {"name": "__import__()", "example": "__import__('math')\n# Output: <module 'math'>", "description": "Imports a module."}
        ]
    }

    with open("README.md", "w") as f:
        f.write("# Python Built-in Functions\n\n")
        for letter, functions in built_in_functions.items():
            f.write(f"## {letter}\n\n")
            for func in functions:
                f.write(f"### {func['name']}\n\n")
                f.write(f"**Description**: {func['description']}\n\n")
                f.write(f"**Example**:\n```python\n{func['example']}\n```\n\n")

if __name__ == "__main__":
    generate_readme()
