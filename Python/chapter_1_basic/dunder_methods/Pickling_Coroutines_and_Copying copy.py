# Python Special Methods (Dunder Methods) Explained

#############################################################################
# PICKLING METHODS
#############################################################################

"""
Pickling is Python's mechanism for serializing and deserializing objects.
These methods control how custom objects are pickled and unpickled.
"""

import pickle
import copy


class PicklingExample:
    """Demonstrates all pickling-related dunder methods."""
    
    def __init__(self, name, value, secret):
        self.name = name          # Regular attribute to be pickled
        self.value = value        # Regular attribute to be pickled
        self.secret = secret      # Sensitive data we don't want to pickle
        self.derived = name * 2   # Derived data that can be recalculated
    
    def __reduce__(self):
        """
        Controls object serialization when pickled.
        
        Returns either:
        - A string (for global objects)
        - A tuple with 2-6 items:
          (callable, args, state, listiteritems, dictitems, state_setter)
        
        Most commonly returns (callable, args, state) where:
        - callable: function that recreates the object
        - args: arguments to pass to the callable
        - state: object's state to restore via __setstate__
        """
        # Return a tuple containing callable and its arguments
        # that will recreate this object, plus optional state
        return (PicklingExample, (self.name, self.value, None), {'derived': self.derived})
    
    def __reduce_ex__(self, protocol):
        """
        Version-specific alternative to __reduce__.
        Takes protocol version as argument (0-5).
        Higher protocol versions are more efficient.
        
        If both __reduce__ and __reduce_ex__ exist, __reduce_ex__ is preferred.
        """
        if protocol >= 2:  # For newer pickle protocols
            # More efficient serialization for newer protocols
            return (PicklingExample, (self.name, self.value, None), {'derived': self.derived})
        else:  # Fallback for older protocols
            return self.__reduce__()
    
    def __getnewargs__(self):
        """
        Used with protocol 2+ to provide arguments for __new__ during unpickling.
        Rarely needed directly as __reduce__ usually handles this.
        
        Returns a tuple of args that will be passed to __new__.
        """
        # Return arguments that will be passed to __new__
        return (self.name, self.value, None)
    
    def __getnewargs_ex__(self):
        """
        Used with protocol 4+ to provide positional and keyword arguments
        for __new__ during unpickling.
        
        Returns a tuple of (args, kwargs) for __new__.
        """
        # Return (args, kwargs) for __new__
        return ((self.name, self.value), {'secret': None})
    
    def __getstate__(self):
        """
        Controls what state gets pickled.
        
        Returns a dictionary or any picklable object representing state.
        This state will be passed to __setstate__ during unpickling.
        """
        # Create a copy of the instance dictionary
        state = self.__dict__.copy()
        
        # Remove sensitive data from what gets pickled
        state['secret'] = None
        
        # We could also remove derived data that can be recalculated
        # del state['derived']
        
        return state
    
    def __setstate__(self, state):
        """
        Controls how state is restored during unpickling.
        
        Takes the state returned by __getstate__ or the third item
        from __reduce__/__reduce_ex__ and uses it to restore object state.
        """
        # Update instance dictionary with the provided state
        self.__dict__.update(state)
        
        # Recalculate any derived values if needed
        if 'derived' not in state:
            self.derived = self.name * 2


# Example usage:
"""
obj = PicklingExample("test", 42, "sensitive_data")
pickled_data = pickle.dumps(obj)
restored_obj = pickle.loads(pickled_data)
# The restored object won't contain the secret data
"""


#############################################################################
# COROUTINE METHODS
#############################################################################

"""
Coroutine methods enable objects to work with Python's async/await syntax.
These methods let you define custom asynchronous behavior.
"""

import asyncio


class AsyncIterableExample:
    """Demonstrates async iterator dunder methods."""
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.current = start
    
    def __await__(self):
        """
        Makes an object awaitable in async functions.
        
        Must return an iterator that yields None and eventually raises
        StopIteration with a result value. Typically implemented by
        returning a generator created with yield expressions.
        
        This enables: result = await my_object
        """
        async def _await_impl():
            # Simulate some async work
            await asyncio.sleep(0.1)
            return f"Result: {self.start}-{self.end}"
        
        # Return the iterator from _await_impl's __await__
        return _await_impl().__await__()
    
    def __aiter__(self):
        """
        Makes an object an asynchronous iterator.
        
        Return self to indicate this object is its own async iterator.
        This enables: async for item in my_object: ...
        """
        # Reset iteration state
        self.current = self.start
        # Return self as the async iterator
        return self
    
    async def __anext__(self):
        """
        Defines how to get the next item from an async iterator.
        
        Should return the next value or raise StopAsyncIteration.
        Called implicitly in: async for item in my_object: ...
        """
        if self.current >= self.end:
            raise StopAsyncIteration
        
        # Simulate async work for each item
        await asyncio.sleep(0.05)
        value = self.current
        self.current += 1
        return value


# Example usage:
"""
async def main():
    # Using __await__
    counter = AsyncIterableExample(1, 5)
    result = await counter
    print(result)  # "Result: 1-5"
    
    # Using __aiter__ and __anext__
    async for num in AsyncIterableExample(1, 5):
        print(num)  # Will print 1, 2, 3, 4
    
asyncio.run(main())
"""


#############################################################################
# COPYING METHODS
#############################################################################

"""
Copying methods control how objects are copied when using the copy module.
These methods let you customize shallow and deep copying behavior.
"""


class CopyableExample:
    """Demonstrates copy-related dunder methods."""
    
    def __init__(self, name, data, reference):
        self.name = name              # Simple immutable attribute
        self.data = data              # Mutable attribute (e.g., list)
        self.reference = reference    # Reference to external object
        self.expensive = self._compute_expensive()
    
    def _compute_expensive(self):
        """Simulate an expensive operation."""
        return sum(range(100000))
    
    def __copy__(self):
        """
        Controls shallow copying behavior (copy.copy()).
        
        Should return a new instance that's a shallow copy of the original.
        Shallow copies duplicate the object but reference the same nested objects.
        """
        # Create a new instance without calling __init__
        new_obj = self.__class__.__new__(self.__class__)
        
        # Copy the instance dictionary
        new_obj.__dict__.update(self.__dict__)
        
        # Customize copy behavior: make a new copy of data (mutable)
        new_obj.data = self.data.copy()  # Shallow copy of the list
        
        # Reuse expensive computed value
        # new_obj.expensive = self.expensive
        
        return new_obj
    
    def __deepcopy__(self, memo):
        """
        Controls deep copying behavior (copy.deepcopy()).
        
        Should return a new instance that's a deep copy of the original.
        Deep copies duplicate both the object and all objects it references.
        
        The memo parameter is a dictionary used to keep track of objects
        already copied to avoid infinite recursion with circular references.
        """
        # Check memo to avoid copying the same object twice
        if id(self) in memo:
            return memo[id(self)]
        
        # Create a new instance without calling __init__
        new_obj = self.__class__.__new__(self.__class__)
        
        # Add new object to memo to handle circular references
        memo[id(self)] = new_obj
        
        # Deep copy each attribute
        new_obj.name = self.name  # No need to copy immutable strings
        new_obj.data = copy.deepcopy(self.data, memo)  # Deep copy mutable data
        new_obj.reference = copy.deepcopy(self.reference, memo)  # Deep copy reference
        
        # Reuse expensive computed value instead of recomputing
        new_obj.expensive = self.expensive
        
        return new_obj


# Example usage:
"""
original = CopyableExample("original", [1, 2, [3, 4]], {"key": "value"})
shallow = copy.copy(original)
deep = copy.deepcopy(original)

# With shallow copy, modifying nested objects affects original
shallow.data[2].append(5)  # Also modifies original.data[2]

# With deep copy, modifying nested objects doesn't affect original
deep.data[2].append(6)  # Doesn't modify original.data[2]
"""