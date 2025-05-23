# Python Special Methods (Dunder Methods): Container Methods

# --------------------------------------------------------------------------------------
# __len__: Defines behavior for len() function
# --------------------------------------------------------------------------------------
class LenExample:
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        """Returns the length of the data collection."""
        return len(self.data)
        
# Usage:
# my_obj = LenExample([1, 2, 3, 4])
# length = len(my_obj)  # Returns 4


# --------------------------------------------------------------------------------------
# __getitem__: Defines behavior for accessing items using index notation obj[key]
# --------------------------------------------------------------------------------------
class GetItemExample:
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, key):
        """Retrieves the item at the given index/key."""
        if isinstance(key, slice):  # Handle slice objects (e.g., obj[1:5])
            return [self.data[i] for i in range(*key.indices(len(self.data)))]
        return self.data[key]  # Handle direct indexing (e.g., obj[3])

# Usage:
# my_obj = GetItemExample([10, 20, 30, 40, 50])
# value = my_obj[2]  # Returns 30
# slice_values = my_obj[1:4]  # Returns [20, 30, 40]


# --------------------------------------------------------------------------------------
# __setitem__: Defines behavior for assigning values using index notation obj[key] = value
# --------------------------------------------------------------------------------------
class SetItemExample:
    def __init__(self, data):
        self.data = data
    
    def __setitem__(self, key, value):
        """Sets the value at the given index/key."""
        self.data[key] = value

# Usage:
# my_obj = SetItemExample([10, 20, 30, 40])
# my_obj[1] = 25  # Changes list to [10, 25, 30, 40]


# --------------------------------------------------------------------------------------
# __delitem__: Defines behavior for deleting items using del obj[key]
# --------------------------------------------------------------------------------------
class DelItemExample:
    def __init__(self, data):
        self.data = data
    
    def __delitem__(self, key):
        """Deletes the item at the given index/key."""
        del self.data[key]

# Usage:
# my_obj = DelItemExample([10, 20, 30, 40])
# del my_obj[1]  # Changes list to [10, 30, 40]


# --------------------------------------------------------------------------------------
# __iter__: Defines behavior for iteration, making the object iterable
# --------------------------------------------------------------------------------------
class IterExample:
    def __init__(self, data):
        self.data = data
        
    def __iter__(self):
        """Returns an iterator for the data."""
        self.index = 0
        return self
        
    def __next__(self):
        """Returns the next item in the sequence."""
        if self.index < len(self.data):
            value = self.data[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

# Usage:
# my_obj = IterExample([10, 20, 30])
# for val in my_obj:
#     print(val)  # Prints 10, 20, 30


# --------------------------------------------------------------------------------------
# __reversed__: Defines behavior for reversed() function
# --------------------------------------------------------------------------------------
class ReversedExample:
    def __init__(self, data):
        self.data = data
    
    def __reversed__(self):
        """Returns a reverse iterator for the data."""
        return iter(self.data[::-1])

# Usage:
# my_obj = ReversedExample([10, 20, 30])
# for val in reversed(my_obj):
#     print(val)  # Prints 30, 20, 10


# --------------------------------------------------------------------------------------
# __contains__: Defines behavior for the 'in' operator
# --------------------------------------------------------------------------------------
class ContainsExample:
    def __init__(self, data):
        self.data = data
    
    def __contains__(self, item):
        """Returns True if item is in data, False otherwise."""
        return item in self.data

# Usage:
# my_obj = ContainsExample([10, 20, 30])
# result = 20 in my_obj  # Returns True
# result = 50 in my_obj  # Returns False


# --------------------------------------------------------------------------------------
# __missing__: Defines behavior when a key is not found in a dictionary-like object
# --------------------------------------------------------------------------------------
class DefaultDict(dict):
    def __init__(self, default_factory=None):
        self.default_factory = default_factory
        super().__init__()
    
    def __missing__(self, key):
        """Called when a key is not found in the dictionary.
        Only works in subclasses of dict."""
        if self.default_factory is None:
            raise KeyError(key)
        value = self.default_factory()
        self[key] = value
        return value

# Usage:
# dd = DefaultDict(list)
# dd["key"].append(1)  # Creates a list at "key" and appends 1
# print(dd)  # {'key': [1]}