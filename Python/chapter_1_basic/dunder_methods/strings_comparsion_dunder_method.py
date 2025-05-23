# Python special methods (dunder methods)
"""
Python special methods, also known as dunder methods (double underscore methods),
are methods that start and end with double underscores (e.g., __init__, __str__).
These methods are not meant to be called directly by the user but are invoked
implicitly by Python in response to certain operations or syntax. They allow
user-defined classes to behave like built-in types and integrate seamlessly
with the Python language.

This file focuses on explaining the following special methods:

String Representation:
    - __str__:  For creating a user-friendly string representation of an object.
    - __repr__: For creating an unambiguous string representation of an object,
                often used for debugging and development.
    - __format__: For customizing string formatting.
    - __bytes__:  For providing a byte representation of an object.

Comparison:
    - __eq__:   For defining equality (==) behavior.
    - __ne__:   For defining inequality (!=) behavior.
    - __lt__:   For defining less than (<) behavior.
    - __le__:   For defining less than or equal to (<=) behavior.
    - __gt__:   For defining greater than (>) behavior.
    - __ge__:   For defining greater than or equal to (>=) behavior.
    - __hash__: For making objects hashable, required for use in sets and dictionaries.
    - __bool__: For defining boolean value of an object (used in boolean contexts like if statements).
"""

class Book:
    """
    A class representing a book with title, author, and pages.
    This class demonstrates various dunder methods for string representation and comparison.
    """
    def __init__(self, title, author, pages):
        """
        Constructor for the Book class.
        Initializes a Book object with a title, author, and number of pages.
        """
        self.title = title
        self.author = author
        self.pages = pages

    # -------------------- String Representation Methods --------------------

    def __str__(self):
        """
        __str__(self)

        This method is called by the str() built-in function and by the print statement
        to compute the "informal" or nicely printable string representation of an object.
        If this method is not defined, Python falls back to using __repr__() if available.

        Purpose: To provide a human-readable string representation of the object.
                 This is what you want to see when you print an object or use str().

        Return Value: A string. Must return a string object as the informal, printable
                      representation of the object.

        Example:
        book = Book("The Hitchhiker's Guide to the Galaxy", "Douglas Adams", 224)
        print(str(book))  # Output: Book: The Hitchhiker's Guide to the Galaxy by Douglas Adams
        """
        return f"Book: {self.title} by {self.author}"

    def __repr__(self):
        """
        __repr__(self)

        This method is called by the repr() built-in function to compute the "official"
        string representation of an object. If possible, this should look like a valid
        Python expression that could be used to recreate an object with the same value
        (given a suitable environment). If this is not possible, it should return a
        string of the form <...some useful description...>. This is used in debugging,
        logging, and for developers to get a detailed view of the object.

        Purpose: To provide an unambiguous string representation of the object.
                 Ideally, it should be possible to recreate the object from this string.
                 If not, it should be informative for debugging.

        Return Value: A string. Must return a string object as the canonical representation.

        Example:
        book = Book("The Hitchhiker's Guide to the Galaxy", "Douglas Adams", 224)
        print(repr(book))  # Output: Book('The Hitchhiker\'s Guide to the Galaxy', 'Douglas Adams', 224)
        """
        return f"Book('{self.title}', '{self.author}', {self.pages})"

    def __format__(self, format_spec):
        """
        __format__(self, format_spec)

        This method is called by the format() built-in function, and by extension,
        formatted string literals (f-strings) and the str.format() method, to produce
        a formatted string representation of the object. The format_spec argument is a
        string containing format specifications.

        Purpose: To customize how an object is formatted when used in formatted strings.
                 Allows control over alignment, padding, precision, etc.

        Parameters:
        - format_spec: A string that specifies the desired formatting.

        Return Value: A formatted string representation of the object.

        Example:
        book = Book("The Lord of the Rings", "J.R.R. Tolkien", 1178)
        print(format(book, 't'))      # Custom format 't' for title only
        print(format(book, 'ap'))     # Custom format 'ap' for author and pages
        print(f"{book:tap}")        # Using f-string with custom format
        """
        if format_spec == 't':
            return self.title
        elif format_spec == 'ap':
            return f"{self.author}, {self.pages} pages"
        else:
            # Default to the string representation if format spec is not recognized
            return str(self)

    def __bytes__(self):
        """
        __bytes__(self)

        Called by bytes() to compute the byte-string representation of an object.
        This should return a bytes object.

        Purpose: To provide a byte representation of the object, useful for serialization,
                 network communication, or interacting with systems that require bytes.

        Return Value: A bytes object representing the byte-string version of the object.

        Example:
        book = Book("1984", "George Orwell", 328)
        book_bytes = bytes(book)
        print(book_bytes) # Output: b'Book: 1984 by George Orwell'
        """
        return bytes(str(self), 'utf-8') # Encoding the string representation to bytes

    # -------------------- Comparison Methods --------------------

    def __eq__(self, other):
        """
        __eq__(self, other)

        Called to implement the equality comparison operator (==). Should return True
        if instances are considered equal, False otherwise. By default, it compares
        object identity (is). Usually overridden to compare object values.

        Purpose: Defines how equality between two objects of this class is determined.

        Parameters:
        - other: The other object to compare with.

        Return Value: True if self and other are considered equal, False otherwise.

        Example:
        book1 = Book("Pride and Prejudice", "Jane Austen", 432)
        book2 = Book("Pride and Prejudice", "Jane Austen", 432)
        book3 = Book("Sense and Sensibility", "Jane Austen", 409)
        print(book1 == book2)  # Output: True
        print(book1 == book3)  # Output: False
        """
        if isinstance(other, Book):
            return (self.title == other.title and
                    self.author == other.author and
                    self.pages == other.pages)
        return False # Not equal if 'other' is not a Book instance

    def __ne__(self, other):
        """
        __ne__(self, other)

        Called to implement the inequality comparison operator (!=). Should return True
        if instances are not considered equal, False otherwise. If __eq__() is defined
        and does not raise TypeError, Python will use the result of __eq__(self, other)
        and negate it.

        Purpose: Defines how inequality between two objects of this class is determined.
                 Often, it's the logical negation of __eq__.

        Parameters:
        - other: The other object to compare with.

        Return Value: True if self and other are considered not equal, False otherwise.

        Example:
        book1 = Book("Pride and Prejudice", "Jane Austen", 432)
        book2 = Book("Pride and Prejudice", "Jane Austen", 432)
        book3 = Book("Sense and Sensibility", "Jane Austen", 409)
        print(book1 != book2)  # Output: False
        print(book1 != book3)  # Output: True
        """
        return not (self == other) # Reusing the __eq__ method for efficiency

    def __lt__(self, other):
        """
        __lt__(self, other)

        Called to implement the less-than comparison operator (<). Should return True
        if self is less than other, False otherwise.

        Purpose: Defines the "less than" relationship between two objects of this class.
                 Allows using the < operator.

        Parameters:
        - other: The other object to compare with.

        Return Value: True if self is less than other, False otherwise.

        Example:
        book1 = Book("A", "Author A", 200)
        book2 = Book("B", "Author B", 300)
        book3 = Book("A", "Author A", 150)
        print(book1 < book2)  # Output: True (comparing by title alphabetically)
        print(book3 < book1)  # Output: False (comparing by pages if titles are same)
        """
        if isinstance(other, Book):
            if self.title != other.title:
                return self.title < other.title # Compare by title first
            else:
                return self.pages < other.pages # Then compare by pages if titles are the same
        return NotImplemented # Indicate that comparison with 'other' type is not implemented

    def __le__(self, other):
        """
        __le__(self, other)

        Called to implement the less-than or equal to comparison operator (<=). Should
        return True if self is less than or equal to other, False otherwise.

        Purpose: Defines the "less than or equal to" relationship. Allows using the <= operator.

        Parameters:
        - other: The other object to compare with.

        Return Value: True if self is less than or equal to other, False otherwise.

        Example:
        book1 = Book("A", "Author A", 200)
        book2 = Book("B", "Author B", 300)
        book3 = Book("A", "Author A", 200)
        print(book1 <= book2)  # Output: True
        print(book3 <= book1)  # Output: True (equal)
        """
        return (self < other) or (self == other) # Reusing __lt__ and __eq__

    def __gt__(self, other):
        """
        __gt__(self, other)

        Called to implement the greater-than comparison operator (>). Should return True
        if self is greater than other, False otherwise.

        Purpose: Defines the "greater than" relationship. Allows using the > operator.

        Parameters:
        - other: The other object to compare with.

        Return Value: True if self is greater than other, False otherwise.

        Example:
        book1 = Book("A", "Author A", 200)
        book2 = Book("B", "Author B", 300)
        book3 = Book("A", "Author A", 150)
        print(book2 > book1)  # Output: True
        print(book1 > book3)  # Output: True
        """
        return not (self <= other) # Efficiently using <= to define >

    def __ge__(self, other):
        """
        __ge__(self, other)

        Called to implement the greater-than or equal to comparison operator (>=). Should
        return True if self is greater than or equal to other, False otherwise.

        Purpose: Defines the "greater than or equal to" relationship. Allows using the >= operator.

        Parameters:
        - other: The other object to compare with.

        Return Value: True if self is greater than or equal to other, False otherwise.

        Example:
        book1 = Book("A", "Author A", 200)
        book2 = Book("B", "Author B", 300)
        book3 = Book("A", "Author A", 200)
        print(book2 >= book1)  # Output: True
        print(book1 >= book3)  # Output: True (equal)
        print(book1 >= book2) # Output: False
        """
        return not (self < other) # Efficiently using < to define >=

    def __hash__(self):
        """
        __hash__(self)

        Called by built-in function hash() and for operations on members of hashed
        collections including set, frozenset, and dict. The purpose of __hash__ is
        to return an integer hash value for the object. Objects that compare equal
        should have the same hash value. If a class does not define __eq__() method
        it should not define __hash__() operation either; if it defines __eq__() but
        not __hash__(), its instances will not be usable as items in hashable collections.

        Purpose: To make objects hashable, enabling their use in sets and as dictionary keys.
                 Objects that are equal according to __eq__ must have the same hash value.

        Return Value: An integer representing the hash value of the object.

        Important: If you define __eq__ and the objects are mutable, you should NOT define __hash__,
                   or make sure the hash value does not change when the object is mutated.
                   For immutable objects or when equality is based on immutable attributes,
                   you can define __hash__.

        Example:
        book1 = Book("Pride and Prejudice", "Jane Austen", 432)
        book2 = Book("Pride and Prejudice", "Jane Austen", 432)
        book_set = {book1, book2} # Sets use hash values to store unique elements.
        print(len(book_set))      # Output: 1 (because book1 and book2 are considered equal)
        """
        return hash((self.title, self.author, self.pages)) # Hash based on immutable attributes

    def __bool__(self):
        """
        __bool__(self)

        Called to implement truth value testing and the built-in operation bool();
        should return False or True. When this method is not defined, __len__() is called,
        if it is defined and returns nonzero value, the object is considered true, otherwise false.
        If __len__() is also not defined, all objects are considered true.

        Purpose: To define whether an object should be considered true or false in a boolean context
                 (e.g., in an if statement, while loop, or when using bool()).

        Return Value: True or False.

        Example:
        book1 = Book("To Kill a Mockingbird", "Harper Lee", 281) # Pages > 0
        book2 = Book("Silent Book", "Author X", 0) # Pages == 0
        print(bool(book1)) # Output: True (default behavior - objects are True)
        print(bool(book2)) # Output: True (default behavior - objects are True)

        # Modify __bool__ to make a book 'False' if it has 0 pages
        # (Uncomment below lines in the class definition to see the effect)
        """
        if self.pages > 0:
            return True
        else:
            return False




# Demonstrating String Representation Methods
print("----- String Representation -----")
book_str_repr = Book("The Great Gatsby", "F. Scott Fitzgerald", 180)
book_str_repr_1=Book("elon","musk",500)

print("Using __str__:", str(book_str_repr))
print("Using __repr__:", repr(book_str_repr))
print("Using format with default:", format(book_str_repr))
print("Using format with custom spec 't':", format(book_str_repr, 't'))
print("Using format with custom spec 'ap':", format(book_str_repr, 'ap'))
print("Using format with custom spec 'a':", format(book_str_repr, 'a'))
print("Using bytes:", bytes(book_str_repr))
print("Using ascii:", ascii(book_str_repr))
print("Using repr:", repr(book_str_repr))
print("Using str:", str(book_str_repr))
print("Using format with custom spec 'a':", format(book_str_repr, 'a'))
print("Using format with custom spec 't':", format(book_str_repr, 't'))
print("Using format with custom spec 'ap':", format(book_str_repr, 'ap'))
print("Using  hash:", hash(book_str_repr))
print("Using bool:", bool(book_str_repr))
print("Using less_than:", book_str_repr < book_str_repr_1)
print("Using less_than_or_equal:", book_str_repr <= book_str_repr_1)
print("Using greater_than:", book_str_repr > book_str_repr_1)
print("Using greater_than_or_equal:", book_str_repr >= book_str_repr_1)
print("Using equal:", book_str_repr == book_str_repr_1)
print("Using not_equal:", book_str_repr != book_str_repr_1)





