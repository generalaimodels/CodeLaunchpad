"""
Python Special Methods (Dunder Methods) - Pickling, Coroutines, and Copying
"""
import pickle
import copy
import asyncio

# =====================================================================
# PICKLING
# =====================================================================
# Pickling is Python's serialization mechanism for converting objects to byte streams

class PicklingExample:
    def __init__(self, data, metadata=None, secret=None):
        self.data = data
        self.metadata = metadata or {}
        self.secret = secret
        self.calculated = len(str(self.data)) * 10
    
    def __reduce__(self):
        """
        Controls basic object pickling. Returns a tuple containing:
        - A callable that will recreate the object
        - Arguments to pass to that callable
        
        This is the fundamental serialization interface. When pickle encounters
        an object, it looks for __reduce__ to determine how to convert the object
        to a string. The returned tuple tells pickle how to reconstruct the object.
        """
        return (PicklingExample, (self.data, self.metadata))
    
    def __reduce_ex__(self, protocol):
        """
        Extended version of __reduce__ that supports different pickle protocols.
        Takes precedence over __reduce__ when defined.
        
        Parameters:
            protocol (int): Pickle protocol version (0-5)
        
        Protocol versions:
        - 0: Original ASCII protocol
        - 1: Old binary format
        - 2: Classes, references
        - 3: Bytes objects
        - 4: Large objects
        - 5: Out-of-band data (Python 3.8+)
        """
        if protocol >= 2:
            # For newer protocols, use more advanced features
            return (PicklingExample, (self.data, self.metadata), self.__getstate__())
        else:
            return self.__reduce__()
    
    def __getnewargs__(self):
        """
        Returns arguments to pass to __new__ during unpickling.
        Only used with pickle protocol 2 and above.
        
        This method provides positional arguments for the class constructor
        when an object is unpickled. It's called before __init__.
        """
        return (self.data,)
    
    def __getnewargs_ex__(self):
        """
        Extended version of __getnewargs__ used with protocol 4+.
        Returns (args, kwargs) tuple to pass to __new__.
        
        This allows specifying both positional and keyword arguments
        during unpickling. Useful for classes with complex initialization.
        """
        args = (self.data,)
        kwargs = {'metadata': self.metadata}
        return (args, kwargs)
    
    def __getstate__(self):
        """
        Controls what state gets pickled.
        
        By default, pickle uses object's __dict__, but this method
        lets you customize the pickled state. Useful for:
        - Excluding non-picklable attributes
        - Removing sensitive data
        - Excluding calculated values that can be regenerated
        """
        state = self.__dict__.copy()
        # Don't pickle sensitive or regenerable data
        if 'secret' in state:
            del state['secret']
        if 'calculated' in state:
            del state['calculated']
        return state
    
    def __setstate__(self, state):
        """
        Restores object state during unpickling.
        
        Called after object creation but before __init__.
        This method controls how the pickled state is applied
        to recreate the object. Useful for:
        - Handling version differences in pickled data
        - Regenerating calculated attributes
        - Setting defaults for missing attributes
        """
        # Restore the pickled state
        self.__dict__.update(state)
        
        # Regenerate calculated values
        if 'calculated' not in self.__dict__:
            self.calculated = len(str(self.data)) * 10
        
        # Set defaults for missing attributes
        if 'secret' not in self.__dict__:
            self.secret = None

# =====================================================================
# COROUTINES
# =====================================================================
# Coroutines enable asynchronous programming with async/await syntax

class AsyncIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    async def __anext__(self):
        """
        Returns the next value in an async iteration.
        
        Must return an awaitable object and should raise
        StopAsyncIteration when iteration is complete.
        Called automatically by the 'async for' statement.
        """
        if self.current >= self.end:
            raise StopAsyncIteration
        
        # Simulate async work
        await asyncio.sleep(0.1)
        
        value = self.current
        self.current += 1
        return value

class AsyncIterable:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __aiter__(self):
        """
        Makes this class an async iterable.
        
        Returns an async iterator that implements __anext__.
        Used with 'async for' statements.
        
        Unlike regular iterators, async iterators can suspend
        execution, allowing other tasks to run during iteration.
        """
        return AsyncIterator(self.start, self.end)

class AwaitableObject:
    def __init__(self, value):
        self.value = value
    
    def __await__(self):
        """
        Makes this object awaitable with the 'await' keyword.
        
        Must return an iterator that yields a series of values.
        The last value becomes the result of the await expression.
        
        This enables objects to integrate with async/await syntax
        and participate in cooperative multitasking.
        """
        async def get_value():
            await asyncio.sleep(0.2)  # Simulate async work
            return self.value
        
        # Return the iterator from the coroutine's __await__
        return get_value().__await__()

# =====================================================================
# COPYING
# =====================================================================
# Python's copy module enables shallow and deep copying of objects

class CopyExample:
    def __init__(self, name, data):
        self.name = name
        self.data = data  # Mutable data
        self.reference = None  # Reference to another object
        self.calculated = len(name) * 10
    
    def set_reference(self, obj):
        self.reference = obj
    
    def __copy__(self):
        """
        Controls shallow copy behavior (copy.copy()).
        
        A shallow copy creates a new object but inserts references
        to the objects found in the original. Mutable objects like
        lists should typically be copied to avoid shared state.
        
        Returns a new instance with copied attributes.
        """
        # Create a new instance
        new_instance = CopyExample(self.name, self.data.copy())
        
        # Copy references as-is (not deeply)
        new_instance.reference = self.reference
        
        # Copy calculated values
        new_instance.calculated = self.calculated
        
        return new_instance
    
    def __deepcopy__(self, memo):
        """
        Controls deep copy behavior (copy.deepcopy()).
        
        A deep copy creates a new object and recursively copies all
        objects found in the original. The memo parameter tracks 
        already-copied objects to handle circular references.
        
        Parameters:
            memo (dict): Dictionary mapping id(obj) to copied objects
        """
        # Check if already copied (handles circular references)
        if id(self) in memo:
            return memo[id(self)]
        
        # Create a new instance with deeply copied data
        new_instance = CopyExample(
            self.name,
            copy.deepcopy(self.data, memo)
        )
        
        # Store in memo to handle circular references
        memo[id(self)] = new_instance
        
        # Deep copy references
        if self.reference is not None:
            new_instance.reference = copy.deepcopy(self.reference, memo)
        
        return new_instance

# =====================================================================
# DEMONSTRATION FUNCTIONS
# =====================================================================

def demonstrate_pickling():
    # Create an object with sensitive data
    obj = PicklingExample("important_data", {"created": "today"}, "password123")
    
    # Pickle and unpickle the object
    pickled_data = pickle.dumps(obj, protocol=4)
    unpickled_obj = pickle.loads(pickled_data)
    
    # The unpickled object has data, metadata, but not the secret
    assert unpickled_obj.data == "important_data"
    assert unpickled_obj.metadata == {"created": "today"}
    assert unpickled_obj.secret is None
    assert hasattr(unpickled_obj, 'calculated')  # Regenerated

async def demonstrate_coroutines():
    # Demonstrate __await__
    awaitable = AwaitableObject(42)
    result = await awaitable  # Calls __await__
    assert result == 42
    
    # Demonstrate __aiter__ and __anext__
    async_iterable = AsyncIterable(1, 5)
    results = []
    
    async for item in async_iterable:  # Uses __aiter__ and __anext__
        results.append(item)
    
    assert results == [1, 2, 3, 4]

def demonstrate_copying():
    # Create objects with circular references
    obj1 = CopyExample("original", [1, 2, 3])
    obj2 = CopyExample("reference", [4, 5, 6])
    
    # Create circular reference
    obj1.set_reference(obj2)
    obj2.set_reference(obj1)
    
    # Shallow copy
    shallow_copy = copy.copy(obj1)
    assert shallow_copy is not obj1  # Different object
    assert shallow_copy.data is not obj1.data  # Data copied
    assert shallow_copy.reference is obj1.reference  # Reference identical
    
    # Deep copy
    deep_copy = copy.deepcopy(obj1)
    assert deep_copy is not obj1  # Different object
    assert deep_copy.data is not obj1.data  # Data copied
    assert deep_copy.reference is not obj1.reference  # Reference copied
    assert deep_copy.reference.reference is deep_copy  # Circular reference preserved

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_pickling()
    asyncio.run(demonstrate_coroutines())
    demonstrate_copying()
    print("All demonstrations completed successfully!")