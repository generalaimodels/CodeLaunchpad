"""
Python Special Methods (Dunder Methods) - Advanced Guide
"""

################################################################################
# __call__ : Makes an instance callable like a function
################################################################################

class Calculator:
    """Class demonstrating the __call__ method for creating callable instances."""
    
    def __init__(self, value=0):
        self.value = value
    
    def __call__(self, x, operation='+'):
        """Make Calculator instances callable with operations.
        
        Args:
            x: Value to operate with
            operation: Mathematical operation to perform (default: '+')
            
        Returns:
            Calculator instance with updated value
        """
        if operation == '+':
            self.value += x
        elif operation == '-':
            self.value -= x
        elif operation == '*':
            self.value *= x
        elif operation == '/':
            if x == 0:
                raise ValueError("Division by zero")
            self.value /= x
        return self
    
    def __repr__(self):
        return f"Calculator({self.value})"

# Usage of __call__
calc = Calculator(10)
# calc can be used as a function
result = calc(5, '+')  # Equivalent to calc.__call__(5, '+')
assert calc.value == 15
# Chaining calls is possible
calc(3, '*')(9, '+')  # First multiplies by 3 (45), then adds 9 (54)
assert calc.value == 54

################################################################################
# Attribute Access Methods
################################################################################

class AttributeDemo:
    """Class demonstrating attribute access special methods."""
    
    def __init__(self):
        self._internal_dict = {'default': 'value'}
        # The next line calls __setattr__
        self.normal_attr = "I'm normal"
    
    def __getattr__(self, name):
        """Called when attribute lookup fails (AttributeError would be raised).
        
        Only invoked if the attribute wasn't found through normal mechanisms.
        """
        print(f"__getattr__ called for: {name}")
        # Check if attribute exists in our internal dictionary
        if name in self._internal_dict:
            return self._internal_dict[name]
        # Still raise AttributeError if we can't find it
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __getattribute__(self, name):
        """Called for ALL attribute access, before __getattr__.
        
        Be careful: calling any self.attr inside this method causes infinite recursion!
        Must use object.__getattribute__(self, name) for internal lookups.
        """
        print(f"__getattribute__ called for: {name}")
        # To access self attributes safely, use the object's __getattribute__
        return object.__getattribute__(self, name)
    
    def __setattr__(self, name, value):
        """Called when an attribute is set."""
        print(f"__setattr__ called: {name} = {value}")
        if name.startswith('_'):
            # For internal attributes, use normal behavior
            object.__setattr__(self, name, value)
        else:
            # For other attributes, store in our internal dict
            self._internal_dict[name] = value
    
    def __delattr__(self, name):
        """Called when an attribute is deleted with del."""
        print(f"__delattr__ called for: {name}")
        if name in self._internal_dict:
            del self._internal_dict[name]
        else:
            # For attributes not in our dict, use default behavior
            object.__delattr__(self, name)
    
    def __dir__(self):
        """Controls what dir(obj) returns.
        
        Useful for custom attribute exposure and autocomplete in interactive shells.
        """
        # Get default dir listing
        default_dir = set(object.__dir__(self))
        # Add the keys from our internal dictionary
        custom_attrs = set(self._internal_dict.keys())
        # Return combined attributes
        return sorted(default_dir.union(custom_attrs))


# Usage Demonstration
obj = AttributeDemo()

# __getattribute__ is called for all attribute access
x = obj.normal_attr                # __getattribute__ called

# __getattr__ is called only when the attribute is not found
try:
    x = obj.missing_attr           # __getattribute__ called, then __getattr__ called
except AttributeError:
    pass

# Setting attributes calls __setattr__
obj.new_attr = "I'm new"           # __setattr__ called

# The dir() function uses __dir__
attrs = dir(obj)                   # "default", "normal_attr", "new_attr" included

# Deleting an attribute calls __delattr__
del obj.new_attr                   # __delattr__ called

################################################################################
# Implementation Order & Important Notes
################################################################################

# 1. Attribute Access Order:
#    - __getattribute__ is always called first for any attribute access
#    - Only if __getattribute__ raises AttributeError, __getattr__ is called
#    - __getattr__ is the "fallback" method

# 2. __getattribute__ Infinite Recursion Danger:
#    - In __getattribute__, any reference to self.attr causes another __getattribute__ call
#    - Always use object.__getattribute__(self, name) for internal lookups

# 3. __setattr__ Auto-invocation:
#    - All attribute assignments (self.attr = value) in __init__ or anywhere call __setattr__
#    - Using self.__dict__[name] = value in __setattr__ causes infinite recursion
#    - Use object.__setattr__(self, name, value) instead

# 4. Use Cases:
#    - __call__: Creating function-like objects, DSLs, builders, currying
#    - Attribute methods: Proxy objects, lazy loading, validation, logging access