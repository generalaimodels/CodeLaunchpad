# -*- coding: utf-8 -*-
"""
Modules and Packages - Advanced Python Code Organization and Reusability

This script provides an in-depth exploration of Python's module and package systems, crucial for structuring large-scale applications and promoting code reusability.
Designed for advanced Python developers, it delves into the mechanics of module loading, the intricacies of package management, and best practices for creating modular and maintainable codebases.

This rigorous exploration will cover:

    - Modules: Advanced module import behaviors, execution as scripts, search paths, and compiled Python files.
    - Standard Modules: Leveraging Python's extensive standard library, emphasizing efficiency and avoiding redundant implementations.
    - The dir() Function: Mastering introspection with `dir()`, exploring module namespaces, and dynamic attribute discovery.
    - Packages: Constructing packages for namespace management, exploring import variations, intra-package references, and advanced package structures.

Expect a focus on:

    - Module loading and caching mechanisms for optimized performance.
    - Advanced import techniques and namespace manipulation.
    - Best practices for package design and dependency management.
    - Understanding the interplay between modules, packages, and Python's import system.
    - Robust strategies for handling import errors and module resolution issues.
    - Pythonic idioms for modular code organization and reusability.

Let's embark on this advanced journey to master Python's module and package system and elevate your code architecture skills.
"""

################################################################################
# 6. Modules
################################################################################

print("\n--- 6. Modules ---\n")

# Modules in Python are fundamental units of code organization, encapsulating functions, classes, and variables within a single file (typically with a `.py` extension).
# They promote modularity, namespace management, and code reuse.

################################################################################
# 6.1. More on Modules
################################################################################

print("\n--- 6.1. More on Modules ---\n")

# --- Basic Module Import ---
print("\n--- Basic Module Import ---")

# Assume we have a module file named 'my_module.py' in the same directory:
# ```python
# # my_module.py
# def greet(name):
#     return f"Hello, {name} from my_module!"
#
# MODULE_CONSTANT = 123
#
# class MyClass:
#     def __init__(self, value):
#         self.value = value
# ```

import my_module # Imports the module 'my_module'

print(f"Module object: {my_module}, Type: {type(my_module)}")
print(f"Accessing function from module: {my_module.greet('Advanced User')}")
print(f"Accessing constant from module: {my_module.MODULE_CONSTANT}")
instance = my_module.MyClass(456)
print(f"Accessing class from module and creating instance: {instance.value}")

# Namespace management: Module import creates a separate namespace, preventing name clashes.
# You access names within the module using dot notation (e.g., `my_module.greet`).

# --- Different Import Statements ---
print("\n--- Different Import Statements ---")

# 1. import module_name: Imports the entire module, namespace accessed via module name.
#    (Demonstrated above)

# 2. from module_name import name1, name2, ...: Imports specific names directly into the current namespace.
from my_module import greet, MODULE_CONSTANT

print(f"Imported greet function directly: {greet('Directly Imported User')}") # No module prefix needed
print(f"Imported MODULE_CONSTANT directly: {MODULE_CONSTANT}") # No module prefix needed

# 3. from module_name import name as alias: Imports a name with a different alias in the current namespace.
from my_module import MyClass as ModuleClassAlias

instance_alias = ModuleClassAlias(789) # Using alias for MyClass
print(f"Imported MyClass with alias ModuleClassAlias: {instance_alias.value}")

# 4. from module_name import *: Imports all public names defined in the module into the current namespace. (Generally discouraged in large projects due to namespace pollution and reduced readability).
# from my_module import * # Imports greet, MODULE_CONSTANT, MyClass directly (avoid in production for clarity)

# --- Module Loading and Caching ---
print("\n--- Module Loading and Caching ---")
# Python imports modules only *once* per interpreter session. Subsequent import statements for the same module simply return the already loaded module object from a cache (`sys.modules`).
# This optimization avoids redundant loading and initialization, improving performance.

import my_module # First import - module is loaded and executed
import my_module # Second import - module is *not* re-executed; cached module is returned

print(f"Are both 'my_module' imports the same object?: {my_module is my_module}") # True - same module object

# To force a module to be reloaded (rarely needed in typical scenarios, mostly for development/debugging):
import importlib
importlib.reload(my_module) # Forces reload and re-execution of module code. Use with caution as it can have side effects if module maintains state.
print("Module reloaded using importlib.reload()")

################################################################################
# 6.1.1. Executing modules as scripts
################################################################################

print("\n--- 6.1.1. Executing modules as scripts ---\n")

# Every Python module has a special attribute `__name__`.
# When a module is run directly as a script (e.g., `python my_module.py`), its `__name__` is set to `"__main__"`.
# When a module is imported into another module, its `__name__` is set to its module name (e.g., `"my_module"`).
# This allows modules to conditionally execute code only when run as scripts, not when imported.

# Add the following to 'my_module.py':
# ```python
# # ... (previous module content) ...
#
# if __name__ == "__main__":
#     print("This code block will only execute when my_module.py is run as a script.")
#     print(greet("Script User"))
#     instance = MyClass(999)
#     print(f"Instance value in script mode: {instance.value}")
# ```

# Running 'python my_module.py' will execute the code within the `if __name__ == "__main__":` block.
# Importing 'my_module' in another script will *not* execute this block.

print("\n--- Demonstrating __name__ in script vs. import mode ---")
print(f"__name__ when running this script directly: {__name__}") # Will be "__main__" when you run this script itself.

import my_module
print(f"my_module.__name__ when imported: {my_module.__name__}") # Will be "my_module"

# Idiomatic use of `if __name__ == "__main__":` for:
# - Unit tests within the module file.
# - Example usage demonstrations when the module is run as a standalone script.
# - Command-line interface (CLI) definitions for the module.

################################################################################
# 6.1.2. The Module Search Path
################################################################################

print("\n--- 6.1.2. The Module Search Path ---\n")

# When you import a module, Python searches for it in a specific sequence of directories, known as the module search path.
# This path is stored in the `sys.path` list.

import sys
print(f"Module search path (sys.path):\n{sys.path}")

# `sys.path` is initialized from:
# 1. The directory containing the input script (or the current directory if no script is specified).
# 2. PYTHONPATH environment variable (a list of directory names, with syntax dependent on the OS).
# 3. Installation-dependent default paths (typically including standard library directories).

# Modifying sys.path at runtime: You can dynamically add directories to `sys.path` to make modules in those directories importable.
import sys
sys.path.insert(0, '/path/to/your/modules') # Insert at the beginning of the path to prioritize this directory.
# sys.path.append('/another/path/for/modules') # Append to the end of the path.

# Be cautious when modifying `sys.path` programmatically, especially in production environments, as it can affect module resolution and introduce unexpected behavior if not managed carefully.
# For project-specific module organization, consider using packages (explained later) or virtual environments for better isolation and dependency management.

################################################################################
# 6.1.3. “Compiled” Python files
################################################################################

print("\n--- 6.1.3. “Compiled” Python files ---\n")

# To speed up module loading, Python caches the *compiled* bytecode of modules in `.pyc` files (or in `__pycache__` directories in Python 3.2+).
# When a module is imported, Python checks if a compiled version exists and if it's newer than the `.py` source file. If both conditions are met, Python loads the bytecode directly, bypassing the compilation step, which can improve import speed, especially for larger modules.

# - `.pyc` files are created automatically when Python has write permissions in the directory containing the `.py` file.
# - In Python 3.2+, `.pyc` files are stored in `__pycache__` subdirectories, organized by Python version (e.g., `__pycache__/my_module.cpython-39.pyc`).
# - Bytecode compilation is transparent to the user in most cases.

# Bytecode invalidation: Python automatically invalidates and recompiles bytecode when the source `.py` file is modified, or when the Python version changes.
# You can control bytecode generation using command-line flags (e.g., `-O` for optimization, `-B` to prevent `.pyc` creation).

# --- Practical implications ---
print("\n--- Practical implications of compiled Python files ---")
# - First import of a module might be slightly slower due to compilation. Subsequent imports are faster due to bytecode caching.
# - Distributing `.pyc` files alone (without `.py` source) is generally *not* recommended as it can lead to compatibility issues and hinder debugging. Distribute source code (`.py` files) primarily.
# - Bytecode caching is primarily an optimization for import speed and usually doesn't require explicit user intervention.

################################################################################
# 6.2. Standard Modules
################################################################################

print("\n--- 6.2. Standard Modules ---\n")

# Python has a vast standard library, a collection of pre-built modules that provide a wide range of functionalities, from operating system interfaces to networking, data serialization, and more.
# Leveraging standard modules is crucial for efficient Python development, avoiding reinventing the wheel and benefiting from well-tested, optimized code.

# --- Examples of Standard Modules ---
print("\n--- Examples of Standard Modules ---")

# 1. os module: Operating system interface (file system operations, process management, environment variables, etc.).
import os
print(f"Operating system name: {os.name}")
print(f"Current working directory: {os.getcwd()}")

# 2. sys module: System-specific parameters and functions (interpreter information, command-line arguments, standard input/output, module path, etc.).
import sys
print(f"Python version: {sys.version}")
print(f"Command line arguments: {sys.argv}")

# 3. math module: Mathematical functions (trigonometric, logarithmic, exponential, etc.).
import math
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Pi constant: {math.pi}")

# 4. random module: Random number generation.
import random
print(f"Random integer between 1 and 10: {random.randint(1, 10)}")
print(f"Random float between 0 and 1: {random.random()}")

# 5. datetime module: Date and time manipulation.
import datetime
now = datetime.datetime.now()
print(f"Current date and time: {now}")

# 6. json module: JSON encoding and decoding.
import json
data = {'name': 'Advanced User', 'level': 'Expert'}
json_string = json.dumps(data)
print(f"JSON string: {json_string}")

# 7. re module: Regular expression operations.
import re
text = "Python is powerful"
match = re.search(r"power\w+", text)
if match:
    print(f"Regex match found: {match.group()}")

# 8. collections module: Container data types (deque, defaultdict, Counter, etc.).
from collections import Counter
word_counts = Counter("abracadabra")
print(f"Word counts using Counter: {word_counts}")

# --- Benefits of using Standard Modules ---
print("\n--- Benefits of using Standard Modules ---")
# - Reusability: Pre-built, well-tested code components.
# - Efficiency: Often implemented in C for performance.
# - Consistency: Standardized APIs across Python installations.
# - Reduced development time: Focus on application logic, not low-level implementation details.
# - Community support and documentation: Extensive resources available.

# Explore the Python Standard Library documentation (https://docs.python.org/3/library/) to discover the wealth of modules available.

################################################################################
# 6.3. The dir() Function
################################################################################

print("\n--- 6.3. The dir() Function ---\n")

# The `dir()` function is a powerful introspection tool in Python. It returns a list of names defined in a namespace.
# Without arguments, `dir()` lists names in the *current* scope (local namespace).
# With a module object as argument, `dir(module)` lists names defined within that *module's* namespace.

# --- dir() in current scope ---
print("\n--- dir() in current scope ---")
current_variable = "hello"
current_function = lambda x: x * 2

names_in_current_scope = dir() # No argument - current scope
print(f"Names in current scope (dir()):\n{names_in_current_scope}")
# Output will include 'current_variable', 'current_function', 'names_in_current_scope', 'dir', 'print', etc.

# --- dir(module) - Exploring module namespace ---
print("\n--- dir(module) - Exploring module namespace ---")
import math
names_in_math_module = dir(math) # Argument is the 'math' module object
print(f"Names in math module (dir(math)):\n{names_in_math_module}")
# Output will include 'acos', 'asin', 'atan', 'ceil', 'cos', 'pi', 'sqrt', etc. (names defined in 'math' module)

# --- dir() for built-in names ---
print("\n--- dir() for built-in names ---")
import builtins # Built-in namespace is always available
names_in_builtins = dir(builtins) # Names available in the built-in namespace (e.g., 'int', 'str', 'list', 'print', 'len', 'Exception', etc.)
print(f"Names in built-in namespace (dir(builtins)):\n{names_in_builtins}")

# --- Filtering dir() output ---
print("\n--- Filtering dir() output ---")
# Often, you might want to filter the output of `dir()` to see only user-defined names, excluding built-in or special names (starting with underscores).

filtered_math_names = [name for name in dir(math) if not name.startswith('_')] # List comprehension to filter out names starting with '_'
print(f"Filtered names in math module (excluding names starting with '_'):\n{filtered_math_names}")

# --- Use cases for dir() ---
print("\n--- Use cases for dir() ---")
# - Exploring the contents of modules and packages.
# - Introspection: Dynamically discovering available attributes and methods of objects.
# - Debugging: Understanding the namespace and available names at different points in your code.
# - Interactive exploration: Quickly examining modules and objects in the Python interpreter.

################################################################################
# 6.4. Packages
################################################################################

print("\n--- 6.4. Packages ---\n")

# Packages are a way of structuring Python modules by using "dotted module names". They are essentially directories that contain:
# 1. Module files (`.py` files).
# 2. A special file named `__init__.py` (can be empty in many cases, but its presence makes Python treat the directory as a package).
# 3. Subpackages (directories that are also packages).

# Package structure example:
# my_package/
#     __init__.py
#     module1.py
#     subpackage/
#         __init__.py
#         module2.py

# --- Importing Modules from Packages ---
print("\n--- Importing Modules from Packages ---")

# Assume the package structure above.

# 1. import package.module: Imports the submodule 'module1' within 'my_package'. You need to use the full dotted name to access names within 'module1'.
import my_package.module1

print(f"Imported my_package.module1: {my_package.module1}")
print(f"Accessing name from submodule: {my_package.module1.some_function()}") # Assuming 'some_function' is defined in 'module1.py'

# 2. from package import module: Imports the submodule 'module1' directly into the current namespace. You can access names within 'module1' using just 'module1.name'.
from my_package import module1

print(f"Imported module1 from my_package: {module1}")
print(f"Accessing name directly: {module1.another_function()}") # Assuming 'another_function' is in 'module1.py'

# 3. from package.module import name: Imports specific names (e.g., 'some_function') directly from 'my_package.module1' into the current namespace. You can access 'some_function' directly.
from my_package.module1 import specific_function

print(f"Imported specific_function from my_package.module1: {specific_function()}") # Access directly

################################################################################
# 6.4.1. Importing * From a Package
################################################################################

print("\n--- 6.4.1. Importing * From a Package ---\n")

# `from package import *` is generally discouraged for packages (similar to `from module import *` but even more problematic).
# By default, `import *` from a package only imports names defined in the package's `__init__.py` file.
# To control what names are imported when using `from package import *`, you can define a list named `__all__` in the package's `__init__.py` file.

# Example:
# In 'my_package/__init__.py':
# ```python
# __all__ = ['module1', 'subpackage'] # List of submodules and subpackages to be imported by 'from my_package import *'
#
# PACKAGE_CONSTANT = "package_value" # Name defined directly in __init__.py will also be imported by 'from package import *'
# ```

# In another script:
# ```python
# from my_package import * # Now only 'module1' and 'subpackage' (and 'PACKAGE_CONSTANT' if defined in __init__.py) will be imported into the current namespace.
#
# print(module1) # OK - module1 is imported
# print(subpackage) # OK - subpackage is imported
# # print(PACKAGE_CONSTANT) # OK - if defined in __init__.py
# ```

# Without `__all__` in `__init__.py`, `from package import *` does very little (might only import names defined directly in `__init__.py` itself, but not submodules).
# Explicitly listing names in `__all__` is considered good practice when you intend to support `from package import *`, but generally, explicit imports (`import package.module`, `from package import module`) are preferred for clarity and maintainability.

################################################################################
# 6.4.2. Intra-package References
################################################################################

print("\n--- 6.4.2. Intra-package References ---\n")

# When modules within a package need to import other modules within the *same* package, you can use intra-package references (relative imports).
# Relative imports use `from .` and `from ..` syntax to specify imports relative to the current module's location within the package hierarchy.

# Example:
# In 'my_package/subpackage/module2.py':
# ```python
# # my_package/subpackage/module2.py
# from . import module3 # Relative import: Import 'module3.py' from the *same* 'subpackage' directory (assuming module3.py exists there)
# from .. import module1 # Relative import: Import 'module1.py' from the *parent* directory ('my_package')

# def function_in_module2():
#     module1.function_in_module1() # Call function from module1 (imported from parent package)
#     module3.function_in_module3() # Call function from module3 (imported from same subpackage)
# ```

# Types of relative imports:
# - `from .module import name`: Import 'name' from 'module' in the *current* package.
# - `from ..module import name`: Import 'name' from 'module' in the *parent* package.
# - `from ...module import name`: Import 'name' from 'module' in the grandparent package, and so on.

# Absolute vs. Relative Imports:
# - Absolute imports (e.g., `import my_package.module1`) are generally preferred for clarity and robustness, especially for top-level imports.
# - Relative imports are useful for internal package structure and when moving or renaming packages, as they are less sensitive to package location changes.

# --- Handling potential ImportErrors with intra-package references ---
print("\n--- Handling potential ImportErrors with intra-package references ---")
# Incorrect relative import paths can lead to ImportError if the relative path doesn't resolve to a valid module within the package structure.
# Carefully verify relative import paths, especially when restructuring packages.

################################################################################
# 6.4.3. Packages in Multiple Directories
################################################################################

print("\n--- 6.4.3. Packages in Multiple Directories ---\n")

# Namespace packages (available in Python 3.3+) allow a single package to be split across multiple directories.
# Unlike regular packages, namespace packages do *not* require an `__init__.py` file in every package directory.
# When Python encounters a package import, it searches through `sys.path` and merges modules and subpackages found in different directories that share the same package name into a single namespace.

# Example:
# Suppose you have two directories in sys.path:
# directory1/my_namespace_package/module_a.py
# directory2/my_namespace_package/module_b.py

# If both 'directory1' and 'directory2' are in `sys.path`, then:
# import my_namespace_package.module_a # Works - loads module_a from directory1
# import my_namespace_package.module_b # Works - loads module_b from directory2
# import my_namespace_package # Also works - the package 'my_namespace_package' is formed by merging content from both directories.

# Namespace packages are useful for:
# - Distributing large packages across multiple repositories or locations.
# - Dynamically extending packages without modifying the original package structure.
# - Decoupling package components that might be developed independently.

# --- Caveats and Advanced Considerations for Namespace Packages ---
print("\n--- Caveats and Advanced Considerations for Namespace Packages ---")
# - Namespace packages do not have a single physical directory representing the package root, which can make certain operations (like package-level initialization) slightly more complex compared to regular packages.
# - Avoid relying on the presence of `__init__.py` in namespace package directories as it's not required and can change the package type if added.
# - Namespace packages are a more advanced feature and might not be necessary for all projects. Regular packages with well-defined directory structures are often sufficient for most use cases.

print("\n--- End of Modules and Packages ---\n")