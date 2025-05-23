"""
This module demonstrates Python's special (dunder) methods for container types.
We provide detailed explanations, examples, and code implementation for:

    1. __len__      - To get the number of elements in the container.
    2. __getitem__  - To retrieve an element via indexing or key access.
    3. __setitem__  - To assign a value to an element via indexing or key access.
    4. __delitem__  - To delete an element via its index or key.
    5. __iter__     - To return an iterator for traversing the container.
    6. __reversed__ - To return a reverse iterator.
    7. __contains__ - To check membership using the "in" operator.
    8. __missing__  - To handle missing keys in mapping types.

Below, we define two custom classes:
    - CustomList: A list-like container implementing most container dunder methods.
    - CustomDict: A dictionary-like container that demonstrates the __missing__ method.

The code follows PEP-8 standards and includes in-line comments to explain each concept.
"""

# --------------------------
# CustomList Implementation
# --------------------------
class CustomList:
    """
    A custom container that mimics list-like behavior by implementing special methods.
    """

    def __init__(self, initial=None):
        """
        Initialize the custom list.
        :param initial: An optional iterable to initialize the container.
        """
        self._data = list(initial) if initial is not None else []

    def __len__(self):
        """
        Return the number of items in the container.
        Called when len() is invoked on an instance.
        """
        return len(self._data)

    def __getitem__(self, index):
        """
        Retrieve element(s) from the container using the indexing operator [].
        Supports both single index and slicing.
        :param index: An integer index or a slice object.
        """
        # Debug/trace statement can be added here if needed.
        return self._data[index]

    def __setitem__(self, index, value):
        """
        Assign a value to an element in the container using [].
        :param index: An integer index or a slice object.
        :param value: The value to assign.
        """
        self._data[index] = value

    def __delitem__(self, index):
        """
        Delete an element from the container at the specified index.
        :param index: An integer index or a slice object.
        """
        del self._data[index]

    def __iter__(self):
        """
        Return an iterator over the container.
        Called when an iteration is requested (e.g., in a for-loop).
        """
        return iter(self._data)

    def __reversed__(self):
        """
        Return a reverse iterator for the container.
        Called when reversed() is invoked.
        """
        return reversed(self._data)

    def __contains__(self, item):
        """
        Check if an item is present in the container.
        Called when using the 'in' operator.
        :param item: The item to check membership for.
        """
        return item in self._data

    def append(self, item):
        """
        Append an item to the end of the container.
        This method is provided for demonstration purposes.
        :param item: The item to append.
        """
        self._data.append(item)

    def __str__(self):
        """
        Return a user-friendly string representation of the container.
        """
        return str(self._data)

    def __repr__(self):
        """
        Return an unambiguous string representation of the container.
        """
        return f"CustomList({self._data!r})"


# --------------------------
# CustomDict Implementation
# --------------------------
class CustomDict(dict):
    """
    A custom mapping type derived from dict that implements the __missing__ method.
    The __missing__ method is automatically called by the __getitem__ method when
    a key is not found in the dictionary.
    """

    def __missing__(self, key):
        """
        Handle missing keys. In this implementation, if a key is not present,
        we return a default value, such as None, or raise a custom error.
        :param key: The key that was not found.
        """
        # For this example, we simply return a default value.
        # Alternative implementations might raise an error:
        # raise KeyError(f'Key {key} not found')
        return f"<No value for {key}>"

    def __getitem__(self, key):
        """
        Override __getitem__ to use __missing__ when key is absent.
        """
        try:
            return super().__getitem__(key)
        except KeyError:
            # Delegate to __missing__ if key isn't found.
            return self.__missing__(key)


# --------------------------
# Example Usage and Tests
# --------------------------
if __name__ == "__main__":
    # --------------------------
    # Testing CustomList Methods
    # --------------------------
    print("Testing CustomList:")

    # Initialize the custom list with some elements.
    clist = CustomList([1, 2, 3, 4, 5])
    print("Initial CustomList:", clist)

    # __len__: Get the length of the list.
    print("Length using __len__:", len(clist))  # Expected output: 5

    # __getitem__: Retrieve an element by index.
    print("Item at index 2 using __getitem__:", clist[2])  # Expected output: 3

    # __setitem__: Change the value at index 3.
    clist[3] = 40
    print("After __setitem__ at index 3:", clist)

    # __delitem__: Delete the item at index 1.
    del clist[1]
    print("After __delitem__ at index 1:", clist)

    # __iter__: Iterate over the container.
    print("Iterating over CustomList:")
    for item in clist:
        print(item, end=" ")
    print()  # New line

    # __reversed__: Iterate in reverse.
    print("Iterating in reverse using __reversed__:")
    for item in reversed(clist):
        print(item, end=" ")
    print()  # New line

    # __contains__: Check membership.
    print("Checking membership using __contains__:")
    print("40 in clist:", 40 in clist)  # Expected output: True
    print("100 in clist:", 100 in clist)  # Expected output: False

    # --------------------------
    # Testing CustomDict Methods
    # --------------------------
    print("\nTesting CustomDict:")

    # Initialize the custom dictionary with some key-value pairs.
    cdict = CustomDict({"a": 1, "b": 2})
    print("Initial CustomDict:", cdict)

    # __getitem__: Normal key access.
    print("Accessing key 'a':", cdict["a"])  # Expected output: 1

    # __missing__: Accessing a key that doesn't exist.
    print("Accessing missing key 'c':", cdict["c"])  # Expected output: "<No value for c>"
    
    # Demonstrate that updating key 'c' assigns an actual value.
    cdict["c"] = 3
    print("After setting key 'c':", cdict)
    print("Accessing key 'c' again:", cdict["c"])  # Expected output: 3