#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Built-in Functions Tutorial
------------------------------------------
This file explains the following built-in functions in depth:
- help()
- breakpoint()
- vars()
- globals()
- locals()

Each function is covered with detailed explanations, examples, and edge cases.
"""

##############################################################################
# 1. help() - Python's built-in documentation system
##############################################################################

def help_function_tutorial():
    """
    The help() function provides interactive help or documentation for Python objects.
    
    Key features:
    - Accesses Python's built-in documentation system
    - Can be used on modules, classes, functions, methods, keywords, etc.
    - Can be called with or without arguments
    # - Based on docstrings ('Documentation') defined in the code
    
    Usage patterns:
    1. help() - Enter interactive help mode
    2. help(object) - Get help on a specific object
    3. help("topic") - Get help on a topic as a string
    """
    
    # Basic usage - get help on a built-in function
    # help(print)  # Uncomment to see output
    
    # Getting help on modules (must be imported first)
    import math
    # help(math)  # Uncomment to see documentation for the entire math module
    # help(math.sin)  # Uncomment to see help for a specific function in the module
    
    # Getting help on custom functions and classes
    # help(help_function_tutorial)  # Help on this function - uses its docstring
    
    # Getting help on a Python keyword (passed as string)
    # help("if")  # Documentation on the 'if' statement
    # help("for")  # Documentation on the 'for' loop
    
    # Advanced: help on a specific topic
    # help("LISTS")  # Information about list objects
    
    # EXCEPTIONS:
    try:
        help(undefined_variable)  # This will raise a NameError
    except NameError:
        pass  # In real usage, help() needs an object that exists
    
    # Example of poor vs. good docstrings affecting help()
    def poor_function(): pass
    
    def good_function():
        """
        This function demonstrates a proper docstring.
        
        A good docstring should explain:
        - What the function does
        - What parameters it accepts
        - What it returns
        - Any exceptions it might raise
        """
        pass
    
    # help(poor_function)  # Shows minimal information
    # help(good_function)  # Shows comprehensive documentation

    # TIP: You can write rich docstrings using reStructuredText or Markdown
    # syntax for better formatting when documentation is generated.
    
    # If you want to store help text instead of displaying it
    import io
    import sys
    
    original_stdout = sys.stdout
    help_text = io.StringIO()
    sys.stdout = help_text
    help(str)
    sys.stdout = original_stdout
    
    # Now help_text.getvalue() contains the help output as a string
    # print(help_text.getvalue()[:100])  # Print first 100 chars as example


##############################################################################
# 2. breakpoint() - Python's built-in debugging entry point
##############################################################################

def breakpoint_function_tutorial():
    """
    The breakpoint() function (added in Python 3.7) provides an easy way to
    enter the debugger at a specific point in your code.
    
    Key features:
    - Drops you into a debugger (pdb by default)
    - Configurable via PYTHONBREAKPOINT environment variable
    - More flexible than hard-coding debugger calls
    - Can be completely disabled via PYTHONBREAKPOINT=0
    
    When called, breakpoint() activates the debugger, allowing you to:
    - Inspect variables
    - Step through code
    - Evaluate expressions
    - Modify program state
    """
    
    # Basic usage - this will pause execution and enter the debugger
    x = 5
    y = 10
    z = x + y
    
    # breakpoint()  # Uncomment to activate the debugger here
    
    # After entering the debugger, you can:
    # - Type 'n' to execute the next line
    # - Type 'c' to continue execution
    # - Type 'p variable_name' to print a variable's value
    # - Type 'q' to quit the debugger
    
    # Common debugger commands in pdb:
    # h: help
    # n: next line (step over)
    # s: step into a function
    # r: return from current function
    # c: continue execution
    # l: list source code
    # p expr: evaluate and print expression
    # pp expr: pretty-print expression
    # w: where - show call stack
    # q: quit
    
    # Configuring the debugger:
    # 1. Use a different debugger (like IPython's):
    #    export PYTHONBREAKPOINT=IPython.embed
    
    # 2. Disable breakpoints:
    #    export PYTHONBREAKPOINT=0
    
    # 3. Use your own custom function:
    #    export PYTHONBREAKPOINT=module.breakpoint_function
    
    # Using breakpoint() conditionally
    error_condition = False
    if error_condition:
        # breakpoint()  # Only breaks into debugger if error_condition is True
        pass
    
    # Programmatically controlling breakpoint behavior:
    import sys
    
    def custom_breakpoint_hook(*args, **kwargs):
        print("Custom breakpoint triggered!")
        # Optionally still call the original debugger:
        # import pdb; pdb.set_trace()
    
    # Store the original hook
    original_hook = sys.breakpointhook
    
    # Example of replacing the default behavior
    # sys.breakpointhook = custom_breakpoint_hook
    # breakpoint()  # Would call our custom function instead of pdb
    
    # Restore the original behavior
    # sys.breakpointhook = original_hook
    
    # IMPORTANT: breakpoint() is preferred over the older style:
    # import pdb; pdb.set_trace()
    # It's more maintainable and configurable


##############################################################################
# 3. vars() - Inspect object's attributes
##############################################################################

def vars_function_tutorial():
    """
    The vars() function returns the __dict__ attribute of an object,
    which contains the object's attributes as a dictionary.
    
    Key features:
    - With no arguments: works like locals()
    - With an object argument: returns object.__dict__
    - Useful for inspecting object attributes
    - Cannot be used on objects without a __dict__ attribute
    
    Syntax:
        vars([object])
    """
    
    # Case 1: vars() with no arguments - returns local namespace (like locals())
    a = 10
    b = "hello"
    local_vars = vars()  # Returns the local variables in this function
    # print(f"Local vars: {local_vars}")  # Contains 'a', 'b', and others
    
    # Case 2: vars() with a module
    import math
    math_vars = vars(math)  # Returns the attributes of the math module
    # print("pi value:", math_vars['pi'])  # Access the pi constant via the dict
    
    # Case 3: vars() with a class instance
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
            
        def greet(self):
            return f"Hello, my name is {self.name}"
    
    person = Person("Alice", 30)
    person_vars = vars(person)  # Returns {'name': 'Alice', 'age': 30}
    # print(f"Person attributes: {person_vars}")
    
    # Example: Using vars() for dynamic attribute access
    for attr_name, attr_value in vars(person).items():
        pass
        # print(f"{attr_name}: {attr_value}")
    
    # Example: Modifying attributes through the dict
    vars(person)['name'] = "Alicia"  # Changes person.name to "Alicia"
    # print(person.name)  # Prints "Alicia"
    
    # EXCEPTIONS: Objects without __dict__
    try:
        vars(42)  # Will raise TypeError - int has no __dict__ attribute
    except TypeError as e:
        # print(f"Error: {e}")
        pass
        
    # Other built-in types without __dict__:
    problematic_objects = [
        42,               # int
        "string",         # str
        [1, 2, 3],        # list
        {1: 'a', 2: 'b'}, # dict (note: vars can't be used on dict, though dict has attributes)
        (1, 2, 3),        # tuple
        {1, 2, 3},        # set
        True,             # bool
        None,             # NoneType
    ]
    
    # Checking which objects support vars()
    for obj in problematic_objects:
        try:
            vars(obj)
            # print(f"{type(obj).__name__} supports vars()")
        except TypeError:
            # print(f"{type(obj).__name__} does NOT support vars()")
            pass
    
    # Advanced use case: Comparing attributes between objects
    person1 = Person("Bob", 25)
    person2 = Person("Bob", 35)
    
    diff_attrs = {}
    for attr, value in vars(person1).items():
        if vars(person2).get(attr) != value:
            diff_attrs[attr] = (value, vars(person2).get(attr))
    
    # print(f"Different attributes: {diff_attrs}")  # Shows {'age': (25, 35)}
    
    # vs using getattr() for similar functionality
    # The key difference: vars() gives you the whole dictionary at once
    # while getattr() accesses attributes one at a time
    # print(getattr(person, 'name'))  # Prints "Alicia"


##############################################################################
# 4. globals() - Access global namespace
##############################################################################

# Global variable examples
GLOBAL_CONSTANT = 100
global_variable = "I am global"

def globals_function_tutorial():
    """
    The globals() function returns a dictionary representing the current global
    namespace, containing all global variables and their values.
    
    Key features:
    - Returns all global variables as a dictionary
    - Mutable - changes to the returned dict affect the global namespace
    - Available in any scope (unlike locals())
    - Includes imported modules and functions defined at module level
    
    Syntax:
        globals()
    """
    
    # Basic usage - get the global namespace
    all_globals = globals()
    # print(f"There are {len(all_globals)} items in the global namespace")
    
    # Accessing global variables
    # print(f"GLOBAL_CONSTANT: {all_globals['GLOBAL_CONSTANT']}")
    # print(f"global_variable: {all_globals['global_variable']}")
    
    # Modifying global variables through globals()
    globals()['global_variable'] = "I was changed"
    # print(f"Modified global_variable: {global_variable}")
    
    # Adding new global variables
    globals()['new_global'] = "I'm a new global variable"
    # Now new_global exists in the global namespace
    # print(f"New global: {new_global}")  # Uncomment to verify
    
    # Difference between accessing a global directly vs. via globals()
    # Direct access is more readable and slightly faster
    direct = global_variable
    via_globals = globals()['global_variable']
    # Both give the same result, but direct access is preferred for normal use
    
    # Using globals() to dynamically execute code in the global scope
    func_code = """
def dynamically_created_function():
    print("This function was created dynamically!")
"""
    exec(func_code, globals())
    # Now this function exists in the global namespace
    # dynamically_created_function()  # Uncomment to call it
    
    # Detecting if a name exists in the global namespace
    variable_to_check = 'math'
    exists = variable_to_check in globals()
    # print(f"Does '{variable_to_check}' exist in globals? {exists}")
    
    # IMPORTANT: Variables declared inside a function are NOT in globals
    local_var = "I am local"
    # print('local_var' in globals())  # False
    
    # Common use case: Implementing a singleton pattern
    if 'singleton_instance' not in globals():
        class Singleton:
            def __init__(self):
                self.value = 0
            
            def increment(self):
                self.value += 1
                return self.value
        
        globals()['singleton_instance'] = Singleton()
    
    # Now singleton_instance will be created only once
    # print(f"Singleton value: {singleton_instance.increment()}")
    
    # Using globals() to implement module-level operations
    def get_all_functions():
        """Return all functions defined in the global namespace."""
        return {name: obj for name, obj in globals().items() 
                if callable(obj) and not name.startswith('_')}
    
    # functions = get_all_functions()
    # print(f"Found {len(functions)} functions in global namespace")
    
    # CAUTION: Be careful when modifying globals() - you can break things!
    # Avoid code like: globals()['print'] = lambda x: None  # This disables print!


##############################################################################
# 5. locals() - Access local namespace
##############################################################################

def locals_function_tutorial():
    """
    The locals() function returns a dictionary representing the current local
    namespace, containing all local variables and their values.
    
    Key features:
    - Returns current local variables as a dictionary
    - Behavior differs between module level and function level
    - At module level, locals() is the same as globals()
    - In functions, locals() is read-only in CPython (changes may not affect actual variables)
    - Useful for introspection and debugging
    
    Syntax:
        locals()
    """
    
    # Basic usage - declaring some local variables
    x = 10
    y = "local string"
    z = [1, 2, 3]
    
    # Get dictionary of local variables
    local_vars = locals()
    # print(f"Local variables: {local_vars}")  # Shows x, y, z, and other function variables
    
    # Checking for a variable in locals
    if 'x' in locals():
        # print(f"x exists and equals {locals()['x']}")
        pass
    
    # IMPORTANT: Updating locals() dictionary might not affect actual variables
    # This is implementation-dependent behavior in CPython!
    locals()['x'] = 999  # This may not update the actual variable 'x'
    # print(f"x after attempting change via locals(): {x}")  # Often still shows 10
    
    # The proper way to update a local variable is direct assignment
    x = 999
    # print(f"x after direct assignment: {x}")  # Now it's 999
    
    # Demonstrating difference between module level and function level locals()
    
    # Example: Using locals() for string formatting (common use case)
    name = "Alice"
    age = 30
    formatted = "Name: {name}, Age: {age}".format(**locals())
    # print(formatted)  # "Name: Alice, Age: 30"
    
    # Example: Using locals() with exec (for dynamic code execution)
    exec("temp_var = x + 5", globals(), locals())
    # Note: In CPython, the above creates temp_var but it may not be accessible
    # directly in the function scope due to how locals() works
    # print("temp_var exists in locals():", 'temp_var' in locals())  # True
    
    try:
        # print(temp_var)  # This might raise NameError in CPython
        pass
    except NameError:
        # print("Cannot access temp_var directly")
        pass
    
    # Using locals() to pass all variables to a nested function
    def nested_function(**kwargs):
        for key, value in kwargs.items():
            pass
            # print(f"Variable {key} = {value}")
    
    # nested_function(**locals())  # Passes all local variables to nested_function
    
    # Common use case: Creating a template context
    template = "Hello {name}, you are {age} years old."
    context = {k: v for k, v in locals().items() if k in ['name', 'age']}
    result = template.format(**context)
    # print(result)  # "Hello Alice, you are 30 years old."
    
    # IMPORTANT: locals() at the module level
    # At module level, locals() is the same as globals()
    # print("At module level, locals() == globals():", 
    #       locals_module_level == globals_module_level)  # True

# Demonstrating locals() at module level
locals_module_level = locals()
globals_module_level = globals()

# At module level, locals() and globals() are the same dictionary
# print("Module level - locals() is globals():", locals() is globals())  # True

##############################################################################
# Summary and Advanced Usage Examples
##############################################################################

def advanced_combined_examples():
    """
    This section demonstrates advanced usage combining multiple built-in functions.
    """
    
    # Example 1: Dynamic function creation and inspection
    def create_function(name, body):
        """Dynamically create a function and add it to globals"""
        code = f"def {name}():\n    {body}"
        exec(code, globals())
        
        # Show function details
        # print(f"Created function: {name}")
        # print(f"Function code:\n{vars(globals()[name])}")
        # help(globals()[name])
    
    # create_function("say_hello", "return 'Hello, world!'")
    # print(say_hello())  # "Hello, world!"
    
    # Example 2: Using locals() for template rendering
    def render_template(template_string, **variables):
        """Simple template renderer using locals() and format"""
        context = {}
        context.update(variables)
        
        # Add locals() from the caller's frame (advanced technique)
        import inspect
        caller_locals = inspect.currentframe().f_back.f_locals
        context.update(caller_locals)
        
        return template_string.format(**context)
    
    name = "Python"
    version = 3.9
    # result = render_template("Language: {name}, Version: {version}, Extra: {extra}", extra="Built-in Functions")
    # print(result)  # "Language: Python, Version: 3.9, Extra: Built-in Functions"
    
    # Example 3: Debug helpers
    def debug_info():
        """Print debugging information about current context"""
        import inspect
        frame = inspect.currentframe().f_back
        
        # print("\n----- DEBUG INFO -----")
        # print(f"Function: {frame.f_code.co_name}")
        # print(f"Line number: {frame.f_lineno}")
        # print(f"Local variables:")
        for name, value in frame.f_locals.items():
            if not name.startswith('__'):
                # print(f"  {name} = {value}")
                pass
        # print("----------------------\n")
    
    a = 42
    b = "test"
    # debug_info()  # Will print info about this part of the code

# If you run this file directly (not imported), show a simple demo
if __name__ == "__main__":
    # Uncomment any of these functions to run the specific tutorial
    # help_function_tutorial()
    # breakpoint_function_tutorial()
    # vars_function_tutorial()
    # globals_function_tutorial()
    # locals_function_tutorial()
    # advanced_combined_examples()
    
    # To properly see the help tutorial, uncomment this:
    # import os
    # os.system("python -c \"import this_module; help(this_module.help_function_tutorial)\"")
    
    print("Python Built-in Functions Tutorial")
    print("Uncomment the function calls in __main__ to run specific tutorials")