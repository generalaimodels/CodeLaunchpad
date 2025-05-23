#!/usr/bin/env python3


# =============================================================================
#                           DICTIONARY (dict)
# =============================================================================
def explain_dict():
    """
    Demonstrates the usage of the dict type in Python.
    
    A dictionary is an unordered collection of key-value pairs.
    Key aspects:
      - Mutable.
      - Keys must be immutable types (e.g., strings, numbers, tuples with immutable elements).
      - Allows efficient retrieval, insertion, and deletion.
    
    Examples show:
      - Creating a dictionary.
      - Accessing, modifying, and adding elements.
      - Iterating over keys and values.
      - Handling a KeyError exception (when accessing a non-existent key).
    """
    # --- Creating a dictionary using literal syntax ---
    my_dict = {"name": "Alice", "age": 30, "city": "New York"}
    
    # --- Accessing dictionary values ---
    # Using the index operator:
    name = my_dict["name"]
    print("Name:", name)
    
    # --- Modifying dictionary ---
    # Updating an existing key's value:
    my_dict["age"] = 31
    # Adding new key-value pair:
    my_dict["profession"] = "Engineer"
    
    # --- Using get() method ---
    # Returns None if the key doesn't exist, or a default if provided:
    country = my_dict.get("country", "USA (Default)")
    print("Country:", country)
    
    # --- Iterating over dictionary items ---
    print("Dictionary Contents:")
    for key, value in my_dict.items():
        print(f"  {key}: {value}")
    
    # --- Exception handling ---
    # Accessing a non-existent key using indexing raises a KeyError.
    try:
        unknown = my_dict["unknown"]
    except KeyError as e:
        print("Caught KeyError:", e)

# =============================================================================
#                            FLOAT
# =============================================================================
def explain_float():
    """
    Demonstrates the usage of the float type in Python.
    
    Float represents real numbers with a fractional component.
    Key aspects:
      - Floats are stored as binary approximations; some decimals may be imprecise.
      - The float() constructor converts other types (e.g., int, str) to float.
    
    Examples:
      - Converting an integer to a float.
      - Converting valid string representations (including exponential notation).
      - Handling ValueError when conversion fails.
      - Showcasing special values like NaN and Infinity.
    """
    # --- Converting an integer to float ---
    int_number = 5
    float_number = float(int_number)
    print("Float from integer:", float_number)
    
    # --- Converting a properly formatted string to float ---
    str_number = "3.14159"
    pi = float(str_number)
    print("Float from string '3.14159':", pi)
    
    # --- Converting using exponential notation ---
    str_expo = "1e-3"
    exp_value = float(str_expo)
    print("Float from exponential string '1e-3':", exp_value)
    
    # --- Exception handling ---
    # Converting an invalid string to float raises a ValueError.
    try:
        fake_float = float("not_a_float")
    except ValueError as e:
        print("Caught ValueError:", e)
    
    # --- Special float values ---
    nan = float("nan")
    inf = float("inf")
    print("NaN value:", nan, "| Check (nan != nan):", nan != nan)  # NaN is never equal to itself.
    print("Infinite value:", inf)

# =============================================================================
#                             BOOLEAN (bool)
# =============================================================================
def explain_bool():
    """
    Demonstrates the usage of the bool type in Python.
    
    Boolean values represent logical truth: True or False.
    Key aspects:
      - The bool() constructor converts a value to a Boolean based on truth testing.
      - Falsy values include: None, False, 0 (or 0.0), empty containers (e.g., "", (), [], {}).
      - All other values are considered True.
    
    Examples:
      - Using bool() on various types.
      - Understanding the outcome of conversion.
      - Custom objects can override __bool__ (or __len__) to customize truth value logic.
    """
    # --- Direct boolean values ---
    true_val = True
    false_val = False
    print("Direct booleans:", true_val, false_val)
    
    # --- Converting various types using bool() ---
    print("bool(0) =", bool(0))           # False - numeric zero is falsy.
    print("bool(1) =", bool(1))           # True - non-zero numbers are truthy.
    print("bool('') =", bool(""))         # False - empty string is falsy.
    print("bool('Hello') =", bool("Hello"))  # True - non-empty string is truthy.
    print("bool([]) =", bool([]))         # False - empty list is falsy.
    print("bool([1, 2, 3]) =", bool([1, 2, 3]))  # True - non-empty list is truthy.
    
    # --- Custom class example for boolean conversion ---
    class Custom:
        def __bool__(self):
            # Custom objects can define their truth value.
            return False
    
    custom_obj = Custom()
    print("bool(custom_obj) =", bool(custom_obj))  # Expected: False

# =============================================================================
#                                MAIN
# =============================================================================
def main():
    """
    Executes all demonstration functions to elucidate the workings of:
      - dict, float, and bool.
    """
    print("=== Demonstration of dict in Python ===")
    explain_dict()
    
    print("\n=== Demonstration of float in Python ===")
    explain_float()
    
    print("\n=== Demonstration of bool in Python ===")
    explain_bool()

if __name__ == "__main__":
    main()