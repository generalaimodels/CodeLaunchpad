#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Dictionary Methods - Comprehensive Guide
==============================================

This module provides a thorough exploration of Python's built-in dictionary methods.
Each method is exhaustively explained with examples, time complexity analysis,
edge cases, and performance considerations.

Author: Claude
Date: 2025-03-14
"""

import timeit
import sys
from typing import Any, Dict, List, Tuple, TypeVar, Optional, Callable, Union, Iterator
from collections import Counter


def separator(method_name: str) -> None:
    """Print a separator with the method name for better readability when executed."""
    print(f"\n{'=' * 80}\n{method_name.upper()}\n{'=' * 80}")


# =============================================================================
# dict.clear() - Remove all items from the dictionary
# =============================================================================

def demonstrate_clear() -> None:
    """
    dict.clear() - Remove all items from the dictionary
    
    Time Complexity: O(n) - Linear time (proportional to dictionary size)
    Space Complexity: O(1) - In-place operation
    
    Key Points:
    - Removes ALL key-value pairs from the dictionary
    - Modifies the dictionary in-place and returns None
    - Maintains the same dictionary object identity
    - Does not affect other dictionaries that might reference the cleared dictionary
    
    Common Pitfalls:
    - Setting dict = {} creates a new empty dictionary rather than clearing the existing one
      (this can cause issues if other variables reference the original dictionary)
    - Does not affect nested dictionaries that might be contained as values
    """
    separator("dict.clear()")
    
    # Basic usage
    user_data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    print(f"Before clear(): {user_data}")
    user_data.clear()
    print(f"After clear(): {user_data}")
    print(f"Dictionary is now empty: {len(user_data) == 0}")
    
    # Adding items after clearing
    user_data["name"] = "Bob"
    print(f"After adding a new key: {user_data}")
    
    # Comparison with dict = {}
    original = {"a": 1, "b": 2, "c": 3}
    reference = original  # Both variables now reference the same dictionary
    
    # Method 1: Using clear() - maintains identity
    print(f"\nUsing clear() method:")
    print(f"Before: original={original}, reference={reference}, id(original)={id(original)}")
    original.clear()
    print(f"After:  original={original}, reference={reference}, id(original)={id(original)},")
    # Note that both original and reference are now empty dictionaries with the same identity
    
    # Method 2: Using dict = {} - creates new dictionary
    original = {"a": 1, "b": 2, "c": 3}
    reference = original
    print(f"\nUsing assignment (dict = {{}}):")
    print(f"Before: original={original}, reference={reference}, id(original)={id(original)}")
    original = {}  # This creates a new empty dictionary and assigns it to 'original'
    print(f"After:  original={original}, reference={reference}, id(original)={id(original)}")
    # Note that reference still contains the original data because original now points to a different dict
    
    # Effect on nested dictionaries
    nested = {"x": 1, "y": 2}
    container = {"nested": nested, "z": 3}
    print(f"\nNested example - before: {container}")
    nested.clear()
    print(f"After clearing nested: {container}")
    # Note that nested dictionary was cleared but still exists in container
    
    # Performance of clear() vs. new dictionary creation
    big_dict = {i: i*i for i in range(10000)}
    
    def test_clear():
        d = big_dict.copy()
        d.clear()
    
    def test_new_dict():
        d = big_dict.copy()
        d = {}
    
    clear_time = timeit.timeit(test_clear, number=1000)
    new_dict_time = timeit.timeit(test_new_dict, number=1000)
    
    print(f"\nPerformance comparison (1000 operations):")
    print(f"Time for clear(): {clear_time:.6f} seconds")
    print(f"Time for dict = {{}}: {new_dict_time:.6f} seconds")
    print(f"Creating a new empty dictionary is faster, but doesn't update references")


# =============================================================================
# dict.copy() - Create a shallow copy of the dictionary
# =============================================================================

def demonstrate_copy() -> None:
    """
    dict.copy() - Return a shallow copy of the dictionary
    
    Time Complexity: O(n) - Linear time (proportional to dictionary size)
    Space Complexity: O(n) - Creates a new dictionary of same size
    
    Key Points:
    - Creates a new dictionary with the same key-value pairs
    - Returns a shallow copy - doesn't copy nested objects, just references to them
    - Alternative to dict(original_dict) for simple dictionary copying
    - For deep copying (recursive copying of all nested objects), use copy.deepcopy()
    
    Common Pitfalls:
    - Mistaking shallow copy for deep copy (nested mutable objects are shared)
    - Not considering that modifying mutable values affects both dictionaries
    - For deep copying, import copy and use copy.deepcopy(dictionary)
    """
    separator("dict.copy()")
    
    # Basic usage
    original = {"name": "Alice", "age": 30, "scores": [85, 90, 78]}
    copied = original.copy()
    
    print(f"Original dict: {original}")
    print(f"Copied dict: {copied}")
    print(f"Are they the same object? {original is copied}")
    print(f"Original ID: {id(original)}, Copy ID: {id(copied)}")
    
    # Demonstrate independence of top-level keys
    print("\nModifying original dictionary:")
    original["name"] = "Bob"
    original["email"] = "bob@example.com"
    print(f"Original after modification: {original}")
    print(f"Copy after original was modified: {copied}")
    
    # Shallow copy behavior with nested mutable objects
    print("\nShallow copy with nested mutable objects:")
    print(f"Original scores before: {original['scores']}")
    print(f"Copied scores before: {copied['scores']}")
    
    # Modifying a nested object affects both dictionaries
    original["scores"].append(95)
    print(f"Original scores after append: {original['scores']}")
    print(f"Copied scores after original's nested list was modified: {copied['scores']}")
    
    # Complex nested structure
    nested_dict = {
        "user": {"name": "Charlie", "age": 25},
        "items": [{"id": 1, "value": "first"}, {"id": 2, "value": "second"}]
    }
    
    shallow_copy = nested_dict.copy()
    
    # Modify nested dictionary
    nested_dict["user"]["name"] = "David"
    nested_dict["items"][0]["value"] = "modified"
    
    print("\nComplex nested structure after modifying original:")
    print(f"Original: {nested_dict}")
    print(f"Shallow copy: {shallow_copy}")
    
    # How to perform a deep copy (when needed)
    import copy
    print("\nDeep copy comparison:")
    deep_original = {
        "user": {"name": "Charlie", "age": 25},
        "items": [{"id": 1, "value": "first"}, {"id": 2, "value": "second"}]
    }
    deep_copied = copy.deepcopy(deep_original)
    
    deep_original["user"]["name"] = "David"
    deep_original["items"][0]["value"] = "modified"
    
    print(f"Deep original after modification: {deep_original}")
    print(f"Deep copy after original's modification: {deep_copied}")
    
    # Alternative ways to create shallow copies
    print("\nAlternative ways to create shallow copies:")
    method1 = dict(original)  # Using dict constructor
    method2 = {**original}    # Using dictionary unpacking (Python 3.5+)
    
    print(f"Using dict(): {method1}")
    print(f"Using dict unpacking: {method2}")
    
    # Performance comparison
    large_dict = {i: i*i for i in range(10000)}
    
    copy_time = timeit.timeit(lambda: large_dict.copy(), number=100)
    dict_time = timeit.timeit(lambda: dict(large_dict), number=100)
    unpacking_time = timeit.timeit(lambda: {**large_dict}, number=100)
    
    print("\nPerformance comparison for shallow copying (100 operations):")
    print(f"dict.copy(): {copy_time:.6f} seconds")
    print(f"dict(): {dict_time:.6f} seconds")
    print(f"Dictionary unpacking: {unpacking_time:.6f} seconds")


# =============================================================================
# dict.fromkeys() - Create a new dictionary with specified keys
# =============================================================================

def demonstrate_fromkeys() -> None:
    """
    dict.fromkeys(iterable[, value]) - Create a new dictionary with keys from iterable
    
    Time Complexity: O(n) - Linear time (proportional to number of keys)
    Space Complexity: O(n) - Creates a new dictionary with n keys
    
    Key Points:
    - Creates a new dictionary with keys from the iterable
    - All keys are assigned the same value (default is None)
    - Static/class method: called as dict.fromkeys(), not instance.fromkeys()
    - Useful for initializing dictionaries with default values
    
    Common Pitfalls:
    - Sharing mutable default value across all keys (all keys will reference the same object)
    - Not realizing that modifying the shared mutable value affects all keys
    - Using fromkeys with unique default values (requires different approach)
    """
    separator("dict.fromkeys()")
    
    # Basic usage with default value (None)
    keys = ["name", "age", "email"]
    user_dict = dict.fromkeys(keys)
    print(f"Dictionary with default None values: {user_dict}")
    
    # With custom default value
    default_score = 0
    students = ["Alice", "Bob", "Charlie"]
    scores = dict.fromkeys(students, default_score)
    print(f"\nStudent scores initialized to {default_score}: {scores}")
    
    # Using different iterables as keys
    tuple_keys = ("red", "green", "blue")
    color_values = dict.fromkeys(tuple_keys, 0)
    print(f"\nColors from tuple: {color_values}")
    
    # Using range as keys
    numbered = dict.fromkeys(range(1, 6), "placeholder")
    print(f"\nNumbered keys: {numbered}")
    
    # Using string characters as keys
    char_dict = dict.fromkeys("hello", 1)
    print(f"\nCharacters as keys: {char_dict}")  # Note: duplicates are removed (only one 'l')
    
    # PITFALL: Using mutable default value
    print("\nPITFALL: Using mutable default value")
    
    # Example 1: List as default value
    users = ["Alice", "Bob", "Charlie"]
    # All keys will reference the SAME list object
    user_data = dict.fromkeys(users, [])
    print(f"Initial user_data: {user_data}")
    
    # Modifying the list for one key affects all keys
    user_data["Alice"].append("Data for Alice")
    print(f"After modifying Alice's data: {user_data}")
    print("Note how all users got the same data because they share the same list object")
    
    # Example 2: Dictionary as default value
    config_keys = ["database", "api", "logging"]
    shared_settings = {"enabled": False}
    config = dict.fromkeys(config_keys, shared_settings)
    print(f"\nInitial config: {config}")
    
    # Modifying the dictionary for one key affects all keys
    config["database"]["enabled"] = True
    print(f"After enabling database: {config}")
    
    # SOLUTION: Using dict comprehension for unique mutable values
    print("\nSOLUTION: Using dict comprehension for unique values")
    
    # Create a dictionary where each key has its own independent list
    better_user_data = {user: [] for user in users}
    print(f"Initial better_user_data: {better_user_data}")
    
    better_user_data["Alice"].append("Data for Alice")
    print(f"After modifying Alice's data: {better_user_data}")
    
    # Similarly for dictionaries
    better_config = {key: {"enabled": False} for key in config_keys}
    print(f"\nInitial better_config: {better_config}")
    
    better_config["database"]["enabled"] = True
    print(f"After enabling database: {better_config}")
    
    # Using custom class/factory function for more complex defaults
    def create_default_config():
        return {"enabled": False, "timeout": 30, "retries": 3}
    
    advanced_config = {key: create_default_config() for key in config_keys}
    print(f"\nAdvanced config with factory function: {advanced_config}")
    
    # Common use case: Counting occurrences
    words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    
    # Method 1: Using fromkeys + update
    count_dict = dict.fromkeys(words, 0)
    for word in words:
        count_dict[word] += 1
    print(f"\nWord counts using fromkeys + update: {count_dict}")
    
    # Method 2: Using Counter (more efficient for this use case)
    count_counter = Counter(words)
    print(f"Word counts using Counter: {count_counter}")
    
    # Performance comparison
    def create_with_fromkeys():
        return dict.fromkeys(range(1000), 0)
    
    def create_with_comprehension():
        return {i: 0 for i in range(1000)}
    
    fromkeys_time = timeit.timeit(create_with_fromkeys, number=1000)
    comprehension_time = timeit.timeit(create_with_comprehension, number=1000)
    
    print("\nPerformance comparison (1000 operations):")
    print(f"dict.fromkeys(): {fromkeys_time:.6f} seconds")
    print(f"Dict comprehension: {comprehension_time:.6f} seconds")


# =============================================================================
# dict.get() - Return the value for key if it exists, else default
# =============================================================================

def demonstrate_get() -> None:
    """
    dict.get(key[, default]) - Return the value for key if key is in the dictionary
    
    Time Complexity: O(1) - Average case, O(n) worst case
    Space Complexity: O(1) - Constant space
    
    Key Points:
    - Returns the value for the specified key if the key exists
    - Returns the specified default value if key doesn't exist (default is None)
    - Never raises a KeyError, unlike direct dictionary access with d[key]
    - Does not modify the dictionary
    
    Common Pitfalls:
    - Not providing a default value when one is needed
    - Forgetting that get() doesn't add the key if it doesn't exist
    - Using get() in a loop for the same key (inefficient compared to checking once)
    - Not considering using dict.setdefault() when you want to add missing keys
    """
    separator("dict.get()")
    
    # Basic usage
    user = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    
    # Getting existing keys
    name = user.get("name")
    age = user.get("age")
    print(f"Name: {name}")
    print(f"Age: {age}")
    
    # Getting non-existent key with default value
    address = user.get("address", "No address provided")
    print(f"Address: {address}")
    
    # Getting non-existent key without default (returns None)
    phone = user.get("phone")
    print(f"Phone: {phone}")
    
    # Comparison with direct dictionary access
    print("\nComparison with direct dictionary access:")
    try:
        # This works fine for existing keys
        direct_name = user["name"]
        print(f"Direct access for 'name': {direct_name}")
        
        # This raises KeyError for non-existent keys
        direct_phone = user["phone"]
        print(f"Direct access for 'phone': {direct_phone}")
    except KeyError as e:
        print(f"KeyError occurred: {e}")
    
    # Checking if get() modifies the dictionary
    print("\nDoes get() modify the dictionary?")
    print(f"Before get(): {user}")
    user.get("country", "Unknown")
    print(f"After get() with default: {user}")
    print("Note: get() doesn't add the key-value pair even with a default")
    
    # Conditional operations based on get() result
    print("\nConditional operations with get():")
    email = user.get("email")
    if email:
        print(f"Sending email to: {email}")
    else:
        print("No email available")
    
    # Using get() with complex default values (like functions or calculations)
    def calculate_tax(income):
        return income * 0.2
    
    income = 50000
    tax_rate = user.get("tax_rate", calculate_tax(income))
    print(f"\nCalculated tax: ${tax_rate}")
    
    # Using get() for nested dictionaries
    nested_user = {
        "name": "Bob",
        "profile": {
            "address": {
                "city": "New York",
                "zipcode": "10001"
            }
        }
    }
    
    # Safe access to nested keys
    def nested_get(dictionary, keys, default=None):
        """Safely get a value from nested dictionaries."""
        current = dictionary
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return default
            else:
                return default
        return current
    
    city = nested_get(nested_user, ["profile", "address", "city"])
    country = nested_get(nested_user, ["profile", "address", "country"], "USA")
    non_existent = nested_get(nested_user, ["profile", "contact", "phone"])
    
    print(f"\nNested get - city: {city}")
    print(f"Nested get - country (with default): {country}")
    print(f"Nested get - non-existent path: {non_existent}")
    
    # Common pattern: Counting with get()
    words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    word_counts = {}
    
    for word in words:
        # Get current count (default 0) and increment by 1
        word_counts[word] = word_counts.get(word, 0) + 1
    
    print(f"\nWord counts using get(): {word_counts}")
    
    # Performance: get() vs. try-except
    large_dict = {str(i): i for i in range(1000)}
    
    def using_get():
        # Half of the lookups will find a key, half won't
        for i in range(2000):
            large_dict.get(str(i), 0)
    
    def using_try_except():
        for i in range(2000):
            try:
                result = large_dict[str(i)]
            except KeyError:
                result = 0
    
    get_time = timeit.timeit(using_get, number=100)
    try_except_time = timeit.timeit(using_try_except, number=100)
    
    print("\nPerformance comparison (100 operations with 2000 lookups each):")
    print(f"Using get(): {get_time:.6f} seconds")
    print(f"Using try-except: {try_except_time:.6f} seconds")


# =============================================================================
# dict.items() - Return a view of dictionary's (key, value) pairs
# =============================================================================

def demonstrate_items() -> None:
    """
    dict.items() - Return a view object of the dictionary's items
    
    Time Complexity: O(1) for the method call, O(n) to iterate through all items
    Space Complexity: O(1) for the view object itself
    
    Key Points:
    - Returns a dict_items view object containing (key, value) tuples
    - View objects are dynamic - they reflect changes to the dictionary
    - Can be used directly in loops to iterate through key-value pairs
    - Can be converted to a list or other iterable types if needed
    - Supports membership testing with 'in' operator (checks key-value pairs)
    
    Common Pitfalls:
    - Assuming the view is a list or tuple (it's a special view object)
    - Not realizing that views reflect dictionary changes automatically
    - Inefficiently converting to list when iteration is the only need
    - Using items() unnecessarily when only keys or values are needed
    """
    separator("dict.items()")
    
    # Basic usage
    user = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    items_view = user.items()
    
    print(f"Type of items_view: {type(items_view)}")
    print(f"Items view: {items_view}")
    
    # Converting to list (if needed)
    items_list = list(items_view)
    print(f"\nItems as list: {items_list}")
    print(f"First item: {items_list[0]}")
    
    # Iterating through key-value pairs
    print("\nIterating through key-value pairs:")
    for key, value in user.items():
        print(f"  {key}: {value}")
    
    # Dynamic nature of views
    print("\nDemonstrating dynamic nature of views:")
    print(f"Original view: {items_view}")
    
    # Modify the dictionary
    user["age"] = 31
    user["address"] = "123 Main St"
    print(f"Updated dictionary: {user}")
    print(f"View after dictionary modification: {items_view}")
    
    # Membership testing
    print("\nMembership testing with items view:")
    print(f"('name', 'Alice') in items_view: {('name', 'Alice') in items_view}")
    print(f"('name', 'Bob') in items_view: {('name', 'Bob') in items_view}")
    print(f"'name' in items_view: {'name' in items_view}")  # False - it checks for (key, value) tuples
    
    # Destructuring and unpacking
    print("\nDestructuring and unpacking:")
    # Unpacking items into two tuples
    keys, values = zip(*user.items())
    print(f"Keys from unpacking: {keys}")
    print(f"Values from unpacking: {values}")
    
    # Common operations with items()
    
    # Creating a new dictionary with keys and values swapped
    inverted = {value: key for key, value in user.items()}
    print(f"\nInverted dictionary: {inverted}")
    
    # Filtering items based on conditions
    filtered = {k: v for k, v in user.items() if isinstance(v, str)}
    print(f"Dictionary with only string values: {filtered}")
    
    # Creating a formatted string representation
    formatted = ", ".join(f"{k}={v}" for k, v in user.items())
    print(f"Formatted string: {formatted}")
    
    # Merging dictionaries while handling duplicates
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"b": 20, "c": 30, "d": 40}
    
    # Keep values from dict1 for duplicate keys
    merged1 = {**dict2, **dict1}
    
    # Custom merge logic for duplicate keys
    merged2 = dict(dict1)
    for key, value in dict2.items():
        if key in merged2:
            merged2[key] = [merged2[key], value]  # Store both values in a list
        else:
            merged2[key] = value
    
    print(f"\ndict1: {dict1}")
    print(f"dict2: {dict2}")
    print(f"Merged (keeping dict1 values): {merged1}")
    print(f"Merged (storing both values): {merged2}")
    
    # Performance considerations
    
    # Creating a large dictionary for testing
    large_dict = {str(i): i for i in range(10000)}
    
    # Compare performance of different iteration methods
    def iterate_keys_values():
        result = []
        for key in large_dict:
            result.append((key, large_dict[key]))
        return result
    
    def iterate_items():
        result = []
        for key, value in large_dict.items():
        # for item in large_dict.items():
        #     key, value = item
            result.append((key, value))
        return result
    
    keys_values_time = timeit.timeit(iterate_keys_values, number=100)
    items_time = timeit.timeit(iterate_items, number=100)
    
    print("\nPerformance comparison (100 iterations through 10,000 items):")
    print(f"Using keys + dict access: {keys_values_time:.6f} seconds")
    print(f"Using items(): {items_time:.6f} seconds")


# =============================================================================
# dict.keys() - Return a view of dictionary's keys
# =============================================================================

def demonstrate_keys() -> None:
    """
    dict.keys() - Return a view object of the dictionary's keys
    
    Time Complexity: O(1) for the method call, O(n) to iterate through all keys
    Space Complexity: O(1) for the view object itself
    
    Key Points:
    - Returns a dict_keys view object containing all keys
    - View objects are dynamic - they reflect changes to the dictionary
    - Can be used directly in loops to iterate through keys
    - Supports set-like operations (union, intersection, difference)
    - Supports membership testing with 'in' operator
    
    Common Pitfalls:
    - Assuming the view is a list or tuple (it's a special view object)
    - Not realizing that views reflect dictionary changes automatically
    - Inefficiently converting to list when iteration is the only need
    - Not taking advantage of the set-like operations on key views
    """
    separator("dict.keys()")
    
    # Basic usage
    user = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    keys_view = user.keys()
    
    print(f"Type of keys_view: {type(keys_view)}")
    print(f"Keys view: {keys_view}")
    
    # Converting to list (if needed)
    keys_list = list(keys_view)
    print(f"\nKeys as list: {keys_list}")
    print(f"First key: {keys_list[0]}")
    
    # Iterating through keys
    print("\nIterating through keys:")
    for key in user.keys():
        print(f"  {key}: {user[key]}")
    
    # Note: Iterating through user directly is equivalent
    print("\nIterating through dictionary directly:")
    for key in user:
        print(f"  {key}: {user[key]}")
    
    # Dynamic nature of views
    print("\nDemonstrating dynamic nature of views:")
    print(f"Original view: {keys_view}")
    
    # Modify the dictionary
    user["address"] = "123 Main St"
    del user["age"]
    print(f"Updated dictionary: {user}")
    print(f"View after dictionary modification: {keys_view}")
    
    # Membership testing
    print("\nMembership testing with keys view:")
    print(f"'name' in keys_view: {'name' in keys_view}")
    print(f"'age' in keys_view: {'age' in keys_view}")
    
    # Set-like operations
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"b": 20, "c": 30, "d": 40}
    
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    
    print("\nSet-like operations on key views:")
    # Union (all keys from both)
    union = keys1 | keys2
    print(f"Union: {union}")
    
    # Intersection (keys in both)
    intersection = keys1 & keys2
    print(f"Intersection: {intersection}")
    
    # Difference (keys in dict1 but not in dict2)
    difference = keys1 - keys2
    print(f"Difference (keys1 - keys2): {difference}")
    
    # Symmetric difference (keys in either but not both)
    symmetric_difference = keys1 ^ keys2
    print(f"Symmetric difference: {symmetric_difference}")
    
    # Common operations with keys()
    
    # Checking if all required keys are present
    required_keys = {"name", "email"}
    has_required = required_keys <= user.keys()  # Subset check
    print(f"\nHas all required keys? {has_required}")
    
    # Filtering another dictionary based on keys
    source = {"name": "Bob", "age": 25, "email": "bob@example.com", "phone": "555-1234"}
    filtered = {k: source[k] for k in source.keys() & required_keys}
    print(f"Filtered dictionary with only required keys: {filtered}")
    
    # Removing specific keys
    unwanted_keys = {"address", "phone"}
    filtered2 = {k: source[k] for k in source.keys() - unwanted_keys}
    print(f"Dictionary with unwanted keys removed: {filtered2}")
    
    # Performance considerations
    
    # Creating a large dictionary for testing
    large_dict = {str(i): i for i in range(10000)}
    
    # Compare performance of different methods for key operations
    def check_all_keys_in():
        # Check if all keys 0-999 are in the dictionary
        result = True
        for i in range(1000):
            result = result and (str(i) in large_dict)
        return result
    
    def check_all_keys_in_view():
        # Check if all keys 0-999 are in the dictionary using keys view
        result = True
        keys = large_dict.keys()
        for i in range(1000):
            result = result and (str(i) in keys)
        return result
    
    # Compare different ways to get all keys
    def get_keys_direct():
        return list(large_dict)
    
    def get_keys_method():
        return list(large_dict.keys())
    
    in_dict_time = timeit.timeit(check_all_keys_in, number=100)
    in_view_time = timeit.timeit(check_all_keys_in_view, number=100)
    keys_direct_time = timeit.timeit(get_keys_direct, number=100)
    keys_method_time = timeit.timeit(get_keys_method, number=100)
    
    print("\nPerformance comparison:")
    print(f"Checking keys in dict: {in_dict_time:.6f} seconds")
    print(f"Checking keys in keys view: {in_view_time:.6f} seconds")
    print(f"Getting keys with list(dict): {keys_direct_time:.6f} seconds")
    print(f"Getting keys with list(dict.keys()): {keys_method_time:.6f} seconds")


# =============================================================================
# dict.pop() - Remove specified key and return its value
# =============================================================================

def demonstrate_pop() -> None:
    """
    dict.pop(key[, default]) - Remove specified key and return the corresponding value
    
    Time Complexity: O(1) - Average case, O(n) worst case
    Space Complexity: O(1) - Constant space
    
    Key Points:
    - Removes the key-value pair and returns the value
    - If key is not found, returns the default value if provided
    - Raises KeyError if key is not found and no default is provided
    - Modifies the dictionary in-place
    - Useful for 'extract and remove' operations
    
    Common Pitfalls:
    - Not handling KeyError when key might not exist and no default is provided
    - Confusing with popitem() which removes an arbitrary item
    - Not capturing the returned value when needed
    - Using pop() in a loop over the dictionary's keys (modifies during iteration)
    """
    separator("dict.pop()")
    
    # Basic usage
    user = {"name": "Alice", "age": 30, "email": "alice@example.com", "temp": "delete me"}
    print(f"Original dictionary: {user}")
    
    # Pop existing key
    temp_value = user.pop("temp")
    print(f"Popped value for 'temp': {temp_value}")
    print(f"Dictionary after pop: {user}")
    
    # Pop with default value (key exists)
    age = user.pop("age", 0)
    print(f"\nPopped value for 'age' with default: {age}")
    print(f"Dictionary after popping 'age': {user}")
    
    # Pop with default value (key doesn't exist)
    phone = user.pop("phone", "No phone")
    print(f"\nPopped value for 'phone' with default: {phone}")
    print(f"Dictionary remains unchanged: {user}")
    
    # Pop without default (key doesn't exist)
    try:
        user.pop("address")
    except KeyError as e:
        print(f"\nKeyError when popping non-existent key without default: {e}")
    
    # Safe pop function
    def safe_pop(dictionary, key, default=None):
        try:
            return dictionary.pop(key)
        except KeyError:
            return default
    
    # Using safe_pop
    address = safe_pop(user, "address", "Unknown")
    print(f"\nSafely popped 'address': {address}")
    
    # Common use cases for pop
    
    # 1. Extract and process values one by one
    queue = {"task1": "Process data", "task2": "Generate report", "task3": "Send email"}
    print(f"\nTask queue: {queue}")
    
    # Process tasks one by one
    while queue:
        # Get the first task (in Python 3.7+ dictionaries maintain insertion order)
        task_id = next(iter(queue))
        task_description = queue.pop(task_id)
        print(f"Processing {task_id}: {task_description}")
    
    print(f"Queue after processing: {queue}")
    
    # 2. Removing and converting keys
    data = {"name": "Bob", "AGE": 25, "Email": "bob@example.com"}
    
    # Normalize keys to lowercase
    normalized = {}
    for key in list(data.keys()):  # Create a list to avoid modification during iteration
        value = data.pop(key)
        normalized[key.lower()] = value
    
    print(f"\nOriginal data: {data}")  # Now empty
    print(f"Normalized data: {normalized}")
    
    # 3. Conditionally remove items
    inventory = {
        "apple": 5,
        "banana": 0,
        "orange": 3,
        "grape": 0
    }
    print(f"\nOriginal inventory: {inventory}")
    
    # Remove out-of-stock items
    out_of_stock = []
    for item in list(inventory.keys()):
        if inventory[item] == 0:
            out_of_stock.append(item)
            inventory.pop(item)
    
    print(f"Updated inventory (in stock only): {inventory}")
    print(f"Out of stock items: {out_of_stock}")
    
    # 4. Extracting nested data
    nested = {
        "user": {
            "id": 123,
            "profile": {
                "name": "Charlie",
                "email": "charlie@example.com"
            }
        }
    }
    
    # Extract and remove the user profile
    user_data = nested["user"]
    profile = user_data.pop("profile")
    
    print(f"\nExtracted profile: {profile}")
    print(f"Remaining nested data: {nested}")
    
    # Performance considerations
    large_dict = {str(i): i for i in range(10000)}
    
    def using_pop_with_default():
        d = large_dict.copy()
        for i in range(11000):  # Some will exist, some won't
            d.pop(str(i), None)
    
    def using_try_except():
        d = large_dict.copy()
        for i in range(11000):
            try:
                d.pop(str(i))
            except KeyError:
                pass
    
    def using_conditional():
        d = large_dict.copy()
        for i in range(11000):
            key = str(i)
            if key in d:
                d.pop(key)
    
    pop_default_time = timeit.timeit(using_pop_with_default, number=5)
    try_except_time = timeit.timeit(using_try_except, number=5)
    conditional_time = timeit.timeit(using_conditional, number=5)
    
    print("\nPerformance comparison (5 iterations):")
    print(f"pop() with default: {pop_default_time:.6f} seconds")
    print(f"pop() with try-except: {try_except_time:.6f} seconds")
    print(f"Conditional check + pop(): {conditional_time:.6f} seconds")


# =============================================================================
# dict.popitem() - Remove and return an arbitrary (key, value) pair
# =============================================================================

def demonstrate_popitem() -> None:
    """
    dict.popitem() - Remove and return a (key, value) pair as a 2-tuple
    
    Time Complexity: O(1) - Constant time
    Space Complexity: O(1) - Constant space
    
    Key Points:
    - Removes and returns an arbitrary (key, value) pair as a tuple
    - In Python 3.7+, removes the last inserted item (LIFO order)
    - Raises KeyError if the dictionary is empty
    - Modifies the dictionary in-place
    - Useful for destructively iterating through a dictionary
    
    Common Pitfalls:
    - Assuming specific order before Python 3.7 (was arbitrary)
    - Not handling KeyError when dictionary might be empty
    - Confusing with pop() which removes a specific key
    - Using popitem() when specific removal order is required
    """
    separator("dict.popitem()")
    
    # Basic usage
    user = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    print(f"Original dictionary: {user}")
    
    # Pop an item (last inserted in Python 3.7+)
    item = user.popitem()
    print(f"Popped item: {item}")
    print(f"Dictionary after popitem(): {user}")
    
    # Pop another item
    item = user.popitem()
    print(f"Popped another item: {item}")
    print(f"Dictionary after second popitem(): {user}")
    
    # Handling empty dictionary
    print("\nHandling empty dictionary:")
    try:
        while True:
            item = user.popitem()
            print(f"Popped: {item}")
    except KeyError:
        print("Dictionary is now empty")
    
    # Demonstrating LIFO behavior (Python 3.7+)
    ordered_dict = {}
    ordered_dict["first"] = 1
    ordered_dict["second"] = 2
    ordered_dict["third"] = 3
    
    print(f"\nDictionary with inserted items: {ordered_dict}")
    
    # Items will be popped in reverse insertion order
    print("Popping items one by one:")
    while ordered_dict:
        key, value = ordered_dict.popitem()
        print(f"  Popped: ({key}, {value})")
    
    # Common use cases
    
    # 1. Processing items in LIFO order
    tasks = {}
    tasks["task1"] = "Low priority"
    tasks["task2"] = "Medium priority"
    tasks["task3"] = "High priority"
    
    print(f"\nTask stack: {tasks}")
    print("Processing tasks in LIFO order:")
    
    while tasks:
        task_id, priority = tasks.popitem()
        print(f"  Processing {task_id} ({priority})")
    
    # 2. Safely consuming a dictionary
    def process_items(dictionary):
        results = []
        try:
            while True:
                key, value = dictionary.popitem()
                results.append(f"Processed {key}: {value}")
        except KeyError:
            pass  # Dictionary is empty
        return results
    
    data = {"a": 1, "b": 2, "c": 3}
    processed = process_items(data)
    
    print(f"\nProcessed items: {processed}")
    print(f"Original dictionary after processing: {data}")  # Now empty
    
    # 3. Converting a dictionary to a list of tuples
    original = {"x": 10, "y": 20, "z": 30}
    pairs = []
    
    # Less efficient method using popitem
    copy1 = original.copy()
    while copy1:
        pairs.append(copy1.popitem())
    
    # More efficient method
    pairs_efficient = list(original.items())
    
    print(f"\nDictionary: {original}")
    print(f"As list of tuples (using popitem): {pairs}")
    print(f"As list of tuples (using items()): {pairs_efficient}")
    
    # Performance considerations
    large_dict = {str(i): i for i in range(10000)}
    
    def using_popitem():
        d = large_dict.copy()
        result = []
        try:
            while True:
                result.append(d.popitem())
        except KeyError:
            pass
        return result
    
    def using_items():
        d = large_dict.copy()
        result = list(d.items())
        d.clear()
        return result
    
    popitem_time = timeit.timeit(using_popitem, number=10)
    items_time = timeit.timeit(using_items, number=10)
    
    print("\nPerformance comparison for converting to list of pairs (10 iterations):")
    print(f"Using popitem() in a loop: {popitem_time:.6f} seconds")
    print(f"Using items() + clear(): {items_time:.6f} seconds")


# =============================================================================
# dict.setdefault() - Return value for key, set default if key not present
# =============================================================================

def demonstrate_setdefault() -> None:
    """
    dict.setdefault(key[, default]) - Get value for key, set & return default if key missing
    
    Time Complexity: O(1) - Average case, O(n) worst case
    Space Complexity: O(1) - Constant space, possibly more if new key added
    
    Key Points:
    - Returns the value for key if it exists
    - If key doesn't exist, inserts key with the specified default value and returns it
    - Default value defaults to None if not specified
    - Combines get and add operations atomically
    - Modifies the dictionary if the key doesn't exist
    
    Common Pitfalls:
    - Confusing with get() which doesn't modify the dictionary
    - Using setdefault() with expensive default values (computed even if not used)
    - Creating a mutable default object that might be shared unintentionally
    - Using setdefault() where a simpler approach would suffice
    """
    separator("dict.setdefault()")
    
    # Basic usage
    user = {"name": "Alice", "age": 30}
    print(f"Original dictionary: {user}")
    
    # Getting existing key
    name = user.setdefault("name", "Unknown")
    print(f"Retrieved name: {name}")
    print(f"Dictionary unchanged: {user}")
    
    # Setting default for non-existent key
    email = user.setdefault("email", "no-email@example.com")
    print(f"\nRetrieved/set email: {email}")
    print(f"Dictionary modified: {user}")
    
    # Using default None
    address = user.setdefault("address")  # Default is None
    print(f"\nRetrieved/set address: {address}")
    print(f"Dictionary with None value: {user}")
    
    # Comparison with alternative approaches
    
    # Approach 1: Check and set (two operations)
    inventory = {"apple": 5, "banana": 10}
    print(f"\nInventory: {inventory}")
    
    # Using if + get pattern
    if "orange" not in inventory:
        inventory["orange"] = 0
    orange_count = inventory["orange"]
    print(f"Orange count (using if+get): {orange_count}")
    print(f"Updated inventory: {inventory}")
    
    # Resetting inventory
    inventory = {"apple": 5, "banana": 10}
    
    # Using setdefault (single atomic operation)
    grape_count = inventory.setdefault("grape", 0)
    print(f"Grape count (using setdefault): {grape_count}")
    print(f"Updated inventory: {inventory}")
    
    # Common use cases for setdefault
    
    # 1. Initializing nested collections
    # Example: Grouping words by their first letter
    words = ["apple", "banana", "apricot", "cherry", "blueberry"]
    groups = {}
    
    for word in words:
        first_letter = word[0]
        # Get or create the list for this letter
        letter_group = groups.setdefault(first_letter, [])
        letter_group.append(word)
    
    print(f"\nGrouped words: {groups}")
    
    # Same example using dict.get
    groups_alt = {}
    for word in words:
        first_letter = word[0]
        if first_letter not in groups_alt:
            groups_alt[first_letter] = []
        groups_alt[first_letter].append(word)
    
    print(f"Grouped words (using get): {groups_alt}")
    
    # 2. Building a frequency counter
    text = "to be or not to be that is the question"
    word_counts = {}
    
    for word in text.split():
        word_counts.setdefault(word, 0)
        word_counts[word] += 1
    
    print(f"\nWord frequencies: {word_counts}")
    
    # More elegant with Counter
    from collections import Counter
    word_counts_counter = Counter(text.split())
    print(f"Word frequencies (using Counter): {dict(word_counts_counter)}")
    
    # PITFALL: Using mutable default values
    print("\nPITFALL: Using mutable default objects")
    
    # Problem example
    users_groups = {}
    alice_groups = users_groups.setdefault("alice", [])
    alice_groups.append("admin")
    
    bob_groups = users_groups.setdefault("bob", [])
    bob_groups.append("user")
    
    # Create a fresh empty list for charlie instead of potential reuse
    charlie_groups = users_groups.setdefault("charlie", [])
    charlie_groups.append("guest")
    
    print(f"Users and their groups: {users_groups}")
    
    # PITFALL: Default value is computed even if not used
    print("\nPITFALL: Default value computation")
    
    def expensive_default():
        print("Computing expensive default...")
        # Simulate expensive computation
        result = sum(range(1000000))
        return result
    
    cache = {"cheap_key": "cheap_value"}
    
    # This will compute the expensive default even though the key exists
    print("Using existing key:")
    cheap_value = cache.setdefault("cheap_key", expensive_default())
    print(f"Retrieved value: {cheap_value}")
    
    # Better approach for expensive defaults
    print("\nBetter approach for expensive defaults:")
    if "another_key" not in cache:
        cache["another_key"] = expensive_default()
    else:
        print("Using cached value")
    another_value = cache["another_key"]
    print(f"Retrieved/computed value: {another_value}")
    
    # Performance comparison
    
    # Create dictionaries with 50% of keys already present
    keys = [str(i) for i in range(1000)]
    base_dict = {k: 1 for k in keys[:500]}  # Half the keys are present
    
    def using_setdefault():
        d = base_dict.copy()
        for k in keys:
            value = d.setdefault(k, 0)
            d[k] = value + 1
    
    def using_get():
        d = base_dict.copy()
        for k in keys:
            d[k] = d.get(k, 0) + 1
    
    def using_if_in():
        d = base_dict.copy()
        for k in keys:
            if k not in d:
                d[k] = 0
            d[k] += 1
    
    setdefault_time = timeit.timeit(using_setdefault, number=1000)
    get_time = timeit.timeit(using_get, number=1000)
    if_in_time = timeit.timeit(using_if_in, number=1000)
    
    print("\nPerformance comparison for increment-or-initialize (1000 iterations):")
    print(f"Using setdefault(): {setdefault_time:.6f} seconds")
    print(f"Using get(): {get_time:.6f} seconds")
    print(f"Using if-in check: {if_in_time:.6f} seconds")


# =============================================================================
# dict.update() - Update dictionary with key-value pairs from another
# =============================================================================

def demonstrate_update() -> None:
    """
    dict.update([other]) - Update dictionary with key-value pairs from other
    
    Time Complexity: O(n) - Linear in the size of the argument
    Space Complexity: O(1) - In-place operation
    
    Key Points:
    - Updates the dictionary with key-value pairs from another dictionary/iterable
    - Overwrites existing keys with new values
    - Can accept another dictionary, an iterable of key-value pairs, or keyword arguments
    - Modifies the dictionary in-place and returns None
    - Powerful method with multiple ways to provide update data
    
    Common Pitfalls:
    - Forgetting that update() modifies the original dictionary
    - Not realizing that it overwrites existing keys
    - Using update() with a list of keys (instead of key-value pairs)
    - Not understanding the different argument forms
    """
    separator("dict.update()")
    
    # Basic usage with another dictionary
    user = {"name": "Alice", "age": 30}
    more_info = {"email": "alice@example.com", "age": 31}  # Note: 'age' will be updated
    
    print(f"Original dictionary: {user}")
    print(f"Update source: {more_info}")
    
    user.update(more_info)
    print(f"After update: {user}")
    
    # Using keyword arguments
    user.update(phone="555-1234", address="123 Main St")
    print(f"After update with keywords: {user}")
    
    # Using an iterable of key-value pairs
    user.update([("job", "Engineer"), ("department", "R&D")])
    print(f"After update with list of tuples: {user}")
    
    # Using zip to create key-value pairs
    keys = ["salary", "start_date"]
    values = [75000, "2023-01-15"]
    user.update(zip(keys, values))
    print(f"After update with zip: {user}")
    
    # Return value is None
    result = user.update({"status": "Active"})
    print(f"Return value of update(): {result}")
    print(f"User after update: {user}")
    
    # Multiple argument forms together
    user.update({"manager": "Bob"}, project="Project X", tasks=["Task 1", "Task 2"])
    print(f"After mixed update: {user}")
    
    # Common use cases
    
    # 1. Merging dictionaries
    defaults = {"theme": "light", "font_size": 12, "language": "en"}
    user_prefs = {"theme": "dark", "notifications": True}
    
    config = defaults.copy()  # Start with defaults
    config.update(user_prefs)  # Override with user preferences
    
    print(f"\nDefaults: {defaults}")
    print(f"User preferences: {user_prefs}")
    print(f"Merged config: {config}")
    
    # 2. Conditionally updating
    current = {"a": 1, "b": 2, "c": 3}
    new_values = {"b": 20, "c": 30, "d": 40}
    
    # Only update specific keys
    allowed_keys = {"b", "d"}
    filtered_update = {k: v for k, v in new_values.items() if k in allowed_keys}
    current.update(filtered_update)
    
    print(f"\nAfter conditional update: {current}")
    
    # 3. Nested dictionary updates
    
    # Method 1: Manual approach
    user_profile = {
        "name": "Alice",
        "settings": {
            "theme": "light",
            "notifications": True
        }
    }
    
    settings_update = {
        "theme": "dark",
        "font_size": 14
    }
    
    # Update nested dictionary
    user_profile["settings"].update(settings_update)
    print(f"\nUpdated nested settings: {user_profile}")
    
    # Method 2: Using recursive update function
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                recursive_update(d[k], v)
            else:
                d[k] = v
    
    user_profile = {
        "name": "Bob",
        "settings": {
            "theme": "light",
            "notifications": {
                "email": True,
                "push": False
            }
        }
    }
    
    complex_update = {
        "settings": {
            "font_size": 16,
            "notifications": {
                "push": True,
                "sms": True
            }
        }
    }
    
    recursive_update(user_profile, complex_update)
    print(f"After recursive update: {user_profile}")
    
    # Handling collisions: different strategies
    
    # Strategy 1: Last writer wins (default behavior)
    base = {"a": 1, "b": 2}
    source1 = {"b": 3, "c": 4}
    source2 = {"c": 5, "d": 6}
    
    combined = base.copy()
    combined.update(source1)
    combined.update(source2)
    
    print(f"\nCombining dictionaries (last writer wins):")
    print(f"base: {base}")
    print(f"source1: {source1}")
    print(f"source2: {source2}")
    print(f"combined: {combined}")
    
    # Strategy 2: Keep track of all values
    def update_with_lists(target, source):
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], list):
                    target[key].append(value)
                else:
                    target[key] = [target[key], value]
            else:
                target[key] = value
    
    multi_valued = base.copy()
    update_with_lists(multi_valued, source1)
    update_with_lists(multi_valued, source2)
    
    print(f"\nCombining with value lists:")
    print(f"result: {multi_valued}")
    
    # Performance considerations
    large_dict = {str(i): i for i in range(1000)}
    updates = {str(i): i * 10 for i in range(500, 1500)}  # Overlap with some new keys
    
    def using_update():
        d = large_dict.copy()
        d.update(updates)
        return d
    
    def using_dictionary_comprehension():
        # Equivalent to {**large_dict, **updates}
        return {**large_dict, **updates}
    
    def using_loop():
        d = large_dict.copy()
        for k, v in updates.items():
            d[k] = v
        return d
    
    update_time = timeit.timeit(using_update, number=1000)
    comprehension_time = timeit.timeit(using_dictionary_comprehension, number=1000)
    loop_time = timeit.timeit(using_loop, number=1000)
    
    print("\nPerformance comparison (1000 iterations):")
    print(f"Using update(): {update_time:.6f} seconds")
    print(f"Using dictionary unpacking: {comprehension_time:.6f} seconds")
    print(f"Using explicit loop: {loop_time:.6f} seconds")


# =============================================================================
# dict.values() - Return a view of dictionary's values
# =============================================================================

def demonstrate_values() -> None:
    """
    dict.values() - Return a view object of the dictionary's values
    
    Time Complexity: O(1) for the method call, O(n) to iterate through all values
    Space Complexity: O(1) for the view object itself
    
    Key Points:
    - Returns a dict_values view object containing all values
    - View objects are dynamic - they reflect changes to the dictionary
    - Can be used directly in loops to iterate through values
    - Can contain duplicate values (unlike keys which are unique)
    - Cannot be indexed directly, but can be converted to a list
    
    Common Pitfalls:
    - Assuming values are unique (they can be duplicates)
    - Assuming the view is a list or tuple (it's a special view object)
    - Not realizing that views reflect dictionary changes automatically
    - Inefficiently converting to list when iteration is the only need
    """
    separator("dict.values()")
    
    # Basic usage
    user = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    values_view = user.values()
    
    print(f"Type of values_view: {type(values_view)}")
    print(f"Values view: {values_view}")
    
    # Converting to list (if needed)
    values_list = list(values_view)
    print(f"\nValues as list: {values_list}")
    print(f"First value: {values_list[0]}")
    
    # Iterating through values
    print("\nIterating through values:")
    for value in user.values():
        print(f"  {value}")
    
    # Dynamic nature of views
    print("\nDemonstrating dynamic nature of views:")
    print(f"Original view: {values_view}")
    
    # Modify the dictionary
    user["age"] = 31
    user["address"] = "123 Main St"
    print(f"Updated dictionary: {user}")
    print(f"View after dictionary modification: {values_view}")
    
    # Values can contain duplicates
    scores = {"math": 90, "science": 85, "history": 90, "art": 95}
    score_values = scores.values()
    print(f"\nScores: {scores}")
    print(f"Score values: {score_values}")
    
    # Counting duplicates
    from collections import Counter
    value_counts = Counter(score_values)
    print(f"Value frequency: {value_counts}")
    
    # Duplicate values make it hard to map back to keys
    duplicate_values = {"a": 1, "b": 2, "c": 1, "d": 3}
    print(f"\nDictionary with duplicates: {duplicate_values}")
    dv_values = duplicate_values.values()
    
    # Creating a reverse mapping (only works well if values are unique)
    # When values have duplicates, only one key will be kept
    reverse_simple = {v: k for k, v in duplicate_values.items()}
    print(f"Simple reverse mapping (loses data): {reverse_simple}")
    
    # Better reverse mapping for duplicate values
    reverse_proper = {}
    for k, v in duplicate_values.items():
        if v not in reverse_proper:
            reverse_proper[v] = []
        reverse_proper[v].append(k)
    
    print(f"Proper reverse mapping: {reverse_proper}")
    
    # Common operations with values()
    
    # 1. Statistical operations
    grades = {"Alice": 90, "Bob": 85, "Charlie": 78, "David": 92}
    grade_values = grades.values()
    
    avg_grade = sum(grade_values) / len(grade_values)
    max_grade = max(grade_values)
    min_grade = min(grade_values)
    
    print(f"\nGrades: {grades}")
    print(f"Average grade: {avg_grade:.1f}")
    print(f"Highest grade: {max_grade}")
    print(f"Lowest grade: {min_grade}")
    
    # 2. Checking conditions on all values
    inventory = {"apple": 10, "banana": 5, "cherry": 15}
    
    all_in_stock = all(count > 0 for count in inventory.values())
    any_low_stock = any(count < 10 for count in inventory.values())
    
    print(f"\nInventory: {inventory}")
    print(f"All items in stock? {all_in_stock}")
    print(f"Any low stock items? {any_low_stock}")
    
    # 3. Filtering dictionaries based on values
    stocks = {"AAPL": 150, "MSFT": 250, "GOOG": 2800, "AMZN": 3400}
    
    # Get only high-value stocks (value > 1000)
    high_value = {k: v for k, v in stocks.items() if v > 1000}
    print(f"\nAll stocks: {stocks}")
    print(f"High-value stocks: {high_value}")


if __name__ == "__main__":
    separator("Demonstrate dict.values()")
    demonstrate_clear()
    demonstrate_copy()
    demonstrate_fromkeys()
    demonstrate_get()
    demonstrate_items()
    demonstrate_keys()
    demonstrate_pop()
    demonstrate_popitem()
    demonstrate_setdefault()
    demonstrate_update()
    demonstrate_values()
