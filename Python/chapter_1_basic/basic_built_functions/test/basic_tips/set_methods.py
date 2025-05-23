#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Set Methods - Comprehensive Guide

This file explores all built-in methods of Python's set data structure with
detailed examples, edge cases, and exception handling.

Sets are unordered collections of unique elements. They are mutable,
but can only contain immutable (hashable) elements.
"""


#------------------------------------------------------------------------------
# 1. set.add(element) - Adds an element to the set
#------------------------------------------------------------------------------
def demonstrate_set_add():
    """
    set.add(element) adds a single element to the set.
    - Time Complexity: O(1) average case
    - If element already exists, the set remains unchanged
    - Only hashable (immutable) objects can be added
    """
    print("\n1. SET.ADD() DEMONSTRATION")
    
    # Basic usage
    fruits = {"apple", "banana"}
    print(f"Original set: {fruits}")
    
    fruits.add("orange")
    print(f"After adding 'orange': {fruits}")
    
    # Adding duplicate element (no effect)
    fruits.add("apple")
    print(f"After adding 'apple' again: {fruits}")
    
    # Example of trying to add an unhashable (mutable) element
    try:
        fruits.add(["grape", "kiwi"])
    except TypeError as e:
        print(f"Exception when adding a list: {e}")
    
    # Adding None is allowed
    fruits.add(None)
    print(f"After adding None: {fruits}")
    
    # Adding mixed types is allowed (though not recommended)
    fruits.add(42)
    print(f"After adding an integer: {fruits}")


#------------------------------------------------------------------------------
# 2. set.clear() - Removes all elements from the set
#------------------------------------------------------------------------------
def demonstrate_set_clear():
    """
    set.clear() removes all elements from the set, leaving an empty set.
    - Time Complexity: O(1)
    - Modifies the set in-place
    - Returns None
    """
    print("\n2. SET.CLEAR() DEMONSTRATION")
    
    numbers = {1, 2, 3, 4, 5}
    print(f"Original set: {numbers}")
    
    result = numbers.clear()
    print(f"After clear(): {numbers}")
    print(f"Return value of clear(): {result}")
    
    # Clear on an already empty set (no error)
    empty_set = set()
    empty_set.clear()
    print(f"After clearing an empty set: {empty_set}")


#------------------------------------------------------------------------------
# 3. set.copy() - Returns a shallow copy of the set
#------------------------------------------------------------------------------
def demonstrate_set_copy():
    """
    set.copy() returns a new set with a shallow copy of the original set.
    - Time Complexity: O(n) where n is the number of elements
    - Creates a new set object with the same elements
    - Shallow copy means nested mutable objects are referenced, not copied
    """
    print("\n3. SET.COPY() DEMONSTRATION")
    
    original = {1, 2, 3, (4, 5)}
    print(f"Original set: {original}")
    
    # Creating a copy
    copied = original.copy()
    print(f"Copied set: {copied}")
    print(f"Are they the same object? {original is copied}")
    print(f"Do they contain equal elements? {original == copied}")
    
    # Modifying the original doesn't affect the copy
    original.add(6)
    print(f"Original after modification: {original}")
    print(f"Copy after original was modified: {copied}")
    
    # Alternative ways to copy
    alt_copy = set(original)
    print(f"Alternative copy using set(): {alt_copy}")


#------------------------------------------------------------------------------
# 4. set.difference(other_set, ...) - Returns elements in set but not in others
#------------------------------------------------------------------------------
def demonstrate_set_difference():
    """
    set.difference(*others) returns a new set with elements that are in this set
    but not in the others.
    - Time Complexity: O(len(set) + len(others))
    - Can accept multiple sets as arguments
    - Also available via the '-' operator
    - Returns a new set; doesn't modify the original
    """
    print("\n4. SET.DIFFERENCE() DEMONSTRATION")
    
    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7}
    set_c = {1, 5, 7, 9}
    
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Set C: {set_c}")
    
    # Difference with one other set
    diff_ab = set_a.difference(set_b)
    print(f"A.difference(B): {diff_ab}")
    
    # Using the '-' operator (equivalent to difference with one set)
    diff_ab_operator = set_a - set_b
    print(f"A - B (using operator): {diff_ab_operator}")
    
    # Difference with multiple sets
    diff_abc = set_a.difference(set_b, set_c)
    print(f"A.difference(B, C): {diff_abc}")
    
    # Difference with an empty set
    print(f"A.difference(empty set): {set_a.difference(set())}")
    
    # Difference with itself
    print(f"A.difference(A): {set_a.difference(set_a)}")
    
    # Difference can also take iterables (not just sets)
    print(f"A.difference([4, 5, 6]): {set_a.difference([4, 5, 6])}")


#------------------------------------------------------------------------------
# 5. set.difference_update(other_set, ...) - Removes elements found in others
#------------------------------------------------------------------------------
def demonstrate_set_difference_update():
    """
    set.difference_update(*others) removes all elements of others from this set.
    - Time Complexity: O(len(others))
    - Modifies the set in-place
    - Can accept multiple sets as arguments
    - Also available via the '-=' operator
    - Returns None
    """
    print("\n5. SET.DIFFERENCE_UPDATE() DEMONSTRATION")
    
    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7}
    set_c = {1, 5, 7, 9}
    
    print(f"Original Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Set C: {set_c}")
    
    # Using difference_update with one set
    set_a_copy = set_a.copy()
    result = set_a_copy.difference_update(set_b)
    print(f"After A.difference_update(B): {set_a_copy}")
    print(f"Return value: {result}")
    
    # Using -= operator (equivalent to difference_update with one set)
    set_a_copy = set_a.copy()
    set_a_copy -= set_b
    print(f"After A -= B: {set_a_copy}")
    
    # Multiple sets with difference_update
    set_a_copy = set_a.copy()
    set_a_copy.difference_update(set_b, set_c)
    print(f"After A.difference_update(B, C): {set_a_copy}")
    
    # With an empty set (no change)
    set_a_copy = set_a.copy()
    set_a_copy.difference_update(set())
    print(f"After A.difference_update(empty set): {set_a_copy}")
    
    # With itself (removes all elements)
    set_a_copy = set_a.copy()
    set_a_copy.difference_update(set_a)
    print(f"After A.difference_update(A): {set_a_copy}")
    
    # With non-set iterables
    set_a_copy = set_a.copy()
    set_a_copy.difference_update([4, 5, 6])
    print(f"After A.difference_update([4, 5, 6]): {set_a_copy}")


#------------------------------------------------------------------------------
# 6. set.discard(element) - Removes element from set if present
#------------------------------------------------------------------------------
def demonstrate_set_discard():
    """
    set.discard(element) removes the element from the set if present.
    - Time Complexity: O(1) average case
    - Does NOT raise an error if element is not found
    - Returns None
    """
    print("\n6. SET.DISCARD() DEMONSTRATION")
    
    numbers = {1, 2, 3, 4, 5}
    print(f"Original set: {numbers}")
    
    # Removing an existing element
    result = numbers.discard(3)
    print(f"After discard(3): {numbers}")
    print(f"Return value: {result}")
    
    # Discarding an element that's not in the set (no error)
    numbers.discard(10)
    print(f"After discard(10) (non-existent element): {numbers}")
    
    # Discarding None from a set that doesn't contain it
    numbers.discard(None)
    print(f"After discard(None): {numbers}")
    
    # Trying to discard an unhashable type
    try:
        numbers.discard([1, 2])
    except TypeError as e:
        print(f"Exception when discarding a list: {e}")


#------------------------------------------------------------------------------
# 7. set.intersection(other_set, ...) - Returns elements common to all sets
#------------------------------------------------------------------------------
def demonstrate_set_intersection():
    """
    set.intersection(*others) returns a new set with elements common to the set
    and all others.
    - Time Complexity: O(min(len(set), len(others)))
    - Can accept multiple sets as arguments
    - Also available via the '&' operator
    - Returns a new set; doesn't modify the original
    """
    print("\n7. SET.INTERSECTION() DEMONSTRATION")
    
    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7}
    set_c = {1, 5, 7, 9}
    
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Set C: {set_c}")
    
    # Intersection with one other set
    intersect_ab = set_a.intersection(set_b)
    print(f"A.intersection(B): {intersect_ab}")
    
    # Using the '&' operator (equivalent to intersection with one set)
    intersect_ab_operator = set_a & set_b
    print(f"A & B (using operator): {intersect_ab_operator}")
    
    # Intersection with multiple sets
    intersect_abc = set_a.intersection(set_b, set_c)
    print(f"A.intersection(B, C): {intersect_abc}")
    
    # Intersection with an empty set
    print(f"A.intersection(empty set): {set_a.intersection(set())}")
    
    # Intersection with itself
    print(f"A.intersection(A): {set_a.intersection(set_a)}")
    
    # Intersection can also take iterables (not just sets)
    print(f"A.intersection([4, 5, 6]): {set_a.intersection([4, 5, 6])}")


#------------------------------------------------------------------------------
# 8. set.intersection_update(other_set, ...) - Updates set with intersection
#------------------------------------------------------------------------------
def demonstrate_set_intersection_update():
    """
    set.intersection_update(*others) updates the set, keeping only elements
    found in it and all others.
    - Time Complexity: O(len(set))
    - Modifies the set in-place
    - Can accept multiple sets as arguments
    - Also available via the '&=' operator
    - Returns None
    """
    print("\n8. SET.INTERSECTION_UPDATE() DEMONSTRATION")
    
    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7}
    set_c = {1, 5, 7, 9}
    
    print(f"Original Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Set C: {set_c}")
    
    # Using intersection_update with one set
    set_a_copy = set_a.copy()
    result = set_a_copy.intersection_update(set_b)
    print(f"After A.intersection_update(B): {set_a_copy}")
    print(f"Return value: {result}")
    
    # Using &= operator (equivalent to intersection_update with one set)
    set_a_copy = set_a.copy()
    set_a_copy &= set_b
    print(f"After A &= B: {set_a_copy}")
    
    # Multiple sets with intersection_update
    set_a_copy = set_a.copy()
    set_a_copy.intersection_update(set_b, set_c)
    print(f"After A.intersection_update(B, C): {set_a_copy}")
    
    # With an empty set (removes all elements)
    set_a_copy = set_a.copy()
    set_a_copy.intersection_update(set())
    print(f"After A.intersection_update(empty set): {set_a_copy}")
    
    # With itself (no change)
    set_a_copy = set_a.copy()
    set_a_copy.intersection_update(set_a)
    print(f"After A.intersection_update(A): {set_a_copy}")
    
    # With non-set iterables
    set_a_copy = set_a.copy()
    set_a_copy.intersection_update([4, 5, 6])
    print(f"After A.intersection_update([4, 5, 6]): {set_a_copy}")


#------------------------------------------------------------------------------
# 9. set.isdisjoint(other_set) - Returns True if sets have no common elements
#------------------------------------------------------------------------------
def demonstrate_set_isdisjoint():
    """
    set.isdisjoint(other) returns True if the set has no elements in common with
    other.
    - Time Complexity: O(min(len(set), len(other)))
    - Returns a boolean value
    - Sets are disjoint if their intersection is empty
    """
    print("\n9. SET.ISDISJOINT() DEMONSTRATION")
    
    set_a = {1, 2, 3}
    set_b = {4, 5, 6}
    set_c = {3, 4, 5}
    empty_set = set()
    
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Set C: {set_c}")
    print(f"Empty set: {empty_set}")
    
    # Disjoint sets (no common elements)
    print(f"A.isdisjoint(B): {set_a.isdisjoint(set_b)}")
    
    # Non-disjoint sets (have common elements)
    print(f"A.isdisjoint(C): {set_a.isdisjoint(set_c)}")
    
    # Any set is disjoint with an empty set
    print(f"A.isdisjoint(empty set): {set_a.isdisjoint(empty_set)}")
    print(f"Empty set.isdisjoint(A): {empty_set.isdisjoint(set_a)}")
    
    # A set is not disjoint with itself (unless it's empty)
    print(f"A.isdisjoint(A): {set_a.isdisjoint(set_a)}")
    print(f"Empty set.isdisjoint(Empty set): {empty_set.isdisjoint(empty_set)}")
    
    # Works with any iterable, not just sets
    print(f"A.isdisjoint([7, 8, 9]): {set_a.isdisjoint([7, 8, 9])}")
    print(f"A.isdisjoint([1, 7, 9]): {set_a.isdisjoint([1, 7, 9])}")


#------------------------------------------------------------------------------
# 10. set.issubset(other_set) - Returns True if set is subset of other_set
#------------------------------------------------------------------------------
def demonstrate_set_issubset():
    """
    set.issubset(other) tests whether every element in the set is in other.
    - Time Complexity: O(len(set))
    - Returns a boolean value
    - Also available via the '<=' operator
    - Strict subset can be checked with '<' operator
    """
    print("\n10. SET.ISSUBSET() DEMONSTRATION")
    
    set_a = {1, 2}
    set_b = {1, 2, 3, 4}
    set_c = {1, 2}
    empty_set = set()
    
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Set C: {set_c}")
    print(f"Empty set: {empty_set}")
    
    # Testing proper subset
    print(f"A.issubset(B): {set_a.issubset(set_b)}")
    
    # Using operator <=
    print(f"A <= B: {set_a <= set_b}")
    
    # Equal sets are subsets of each other
    print(f"A.issubset(C): {set_a.issubset(set_c)}")
    print(f"C.issubset(A): {set_c.issubset(set_a)}")
    
    # Strict subset (using < operator)
    print(f"A < B (strict subset): {set_a < set_b}")
    print(f"A < C (equal sets): {set_a < set_c}")
    
    # Empty set is a subset of any set
    print(f"Empty set.issubset(A): {empty_set.issubset(set_a)}")
    
    # Every set is a subset of itself
    print(f"A.issubset(A): {set_a.issubset(set_a)}")
    
    # Works with any iterable
    print(f"A.issubset([1, 2, 3, 4]): {set_a.issubset([1, 2, 3, 4])}")


#------------------------------------------------------------------------------
# 11. set.issuperset(other_set) - Returns True if set is superset of other_set
#------------------------------------------------------------------------------
def demonstrate_set_issuperset():
    """
    set.issuperset(other) tests whether every element in other is in the set.
    - Time Complexity: O(len(other))
    - Returns a boolean value
    - Also available via the '>=' operator
    - Strict superset can be checked with '>' operator
    """
    print("\n11. SET.ISSUPERSET() DEMONSTRATION")
    
    set_a = {1, 2, 3, 4}
    set_b = {1, 2}
    set_c = {1, 2, 3, 4}
    empty_set = set()
    
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Set C: {set_c}")
    print(f"Empty set: {empty_set}")
    
    # Testing proper superset
    print(f"A.issuperset(B): {set_a.issuperset(set_b)}")
    
    # Using operator >=
    print(f"A >= B: {set_a >= set_b}")
    
    # Equal sets are supersets of each other
    print(f"A.issuperset(C): {set_a.issuperset(set_c)}")
    print(f"C.issuperset(A): {set_c.issuperset(set_a)}")
    
    # Strict superset (using > operator)
    print(f"A > B (strict superset): {set_a > set_b}")
    print(f"A > C (equal sets): {set_a > set_c}")
    
    # Any set is a superset of the empty set
    print(f"A.issuperset(empty set): {set_a.issuperset(empty_set)}")
    
    # Empty set is not a superset of non-empty sets
    print(f"Empty set.issuperset(A): {empty_set.issuperset(set_a)}")
    
    # Every set is a superset of itself
    print(f"A.issuperset(A): {set_a.issuperset(set_a)}")
    
    # Works with any iterable
    print(f"A.issuperset([1, 2]): {set_a.issuperset([1, 2])}")


#------------------------------------------------------------------------------
# 12. set.pop() - Removes and returns an arbitrary element from the set
#------------------------------------------------------------------------------
def demonstrate_set_pop():
    """
    set.pop() removes and returns an arbitrary element from the set.
    - Time Complexity: O(1)
    - Raises KeyError if the set is empty
    - Set is unordered, so there's no guarantee which element gets popped
    - Modifies the set in-place
    """
    print("\n12. SET.POP() DEMONSTRATION")
    
    numbers = {1, 2, 3, 4, 5}
    print(f"Original set: {numbers}")
    
    # Popping one element
    popped = numbers.pop()
    print(f"Popped element: {popped}")
    print(f"Set after pop(): {numbers}")
    
    # Popping from a set with one element
    single_set = {42}
    print(f"Single-element set: {single_set}")
    popped = single_set.pop()
    print(f"Popped from single-element set: {popped}")
    print(f"Set is now empty: {single_set}")
    
    # Popping from an empty set raises KeyError
    empty_set = set()
    try:
        empty_set.pop()
    except KeyError as e:
        print(f"Exception when popping from empty set: {e}")
    
    # Popping all elements from a set
    sample_set = {10, 20, 30}
    print(f"Sample set before popping all: {sample_set}")
    while sample_set:
        print(f"Popped: {sample_set.pop()}")
    print(f"Set after popping all elements: {sample_set}")


#------------------------------------------------------------------------------
# 13. set.remove(element) - Removes element from set
#------------------------------------------------------------------------------
def demonstrate_set_remove():
    """
    set.remove(element) removes the element from the set.
    - Time Complexity: O(1) average case
    - Raises KeyError if element is not in the set
    - Modifies the set in-place
    - Returns None
    """
    print("\n13. SET.REMOVE() DEMONSTRATION")
    
    fruits = {"apple", "banana", "orange", "pear"}
    print(f"Original set: {fruits}")
    
    # Removing an existing element
    result = fruits.remove("banana")
    print(f"After remove('banana'): {fruits}")
    print(f"Return value: {result}")
    
    # Attempting to remove a non-existent element
    try:
        fruits.remove("grape")
    except KeyError as e:
        print(f"Exception when removing non-existent element: {e}")
    
    # Comparison with discard() which doesn't raise an error
    print("\nComparison with discard():")
    print(f"Current set: {fruits}")
    fruits.discard("mango")  # No error
    print(f"After discard('mango'): {fruits}")
    
    # Attempting to remove an unhashable type
    try:
        fruits.remove(["apple", "orange"])
    except TypeError as e:
        print(f"Exception when removing unhashable type: {e}")


#------------------------------------------------------------------------------
# 14. set.symmetric_difference(other_set) - Returns elements in either set, but not both
#------------------------------------------------------------------------------
def demonstrate_set_symmetric_difference():
    """
    set.symmetric_difference(other) returns a new set with elements in either
    the set or other but not both.
    - Time Complexity: O(len(set) + len(other))
    - Also available via the '^' operator
    - Returns a new set; doesn't modify the original
    """
    print("\n14. SET.SYMMETRIC_DIFFERENCE() DEMONSTRATION")
    
    set_a = {1, 2, 3, 4}
    set_b = {3, 4, 5, 6}
    
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    
    # Using symmetric_difference method
    sym_diff = set_a.symmetric_difference(set_b)
    print(f"A.symmetric_difference(B): {sym_diff}")
    
    # Using ^ operator
    sym_diff_operator = set_a ^ set_b
    print(f"A ^ B (using operator): {sym_diff_operator}")
    
    # Symmetric difference with an empty set
    print(f"A.symmetric_difference(empty set): {set_a.symmetric_difference(set())}")
    
    # Symmetric difference with itself (always empty)
    print(f"A.symmetric_difference(A): {set_a.symmetric_difference(set_a)}")
    
    # Can take any iterable, not just sets
    print(f"A.symmetric_difference([3, 4, 5, 6]): {set_a.symmetric_difference([3, 4, 5, 6])}")
    
    # Alternative way to calculate symmetric difference
    alt_sym_diff = (set_a - set_b) | (set_b - set_a)
    print(f"Alternative calculation (A-B)|(B-A): {alt_sym_diff}")


if __name__ == "__main__":
    demonstrate_set_pop()
    demonstrate_set_remove()
    demonstrate_set_symmetric_difference()
    # demonstrate_set_union()
    demonstrate_set_intersection()
    demonstrate_set_intersection_update()