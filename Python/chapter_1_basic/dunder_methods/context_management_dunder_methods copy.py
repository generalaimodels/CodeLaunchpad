"""
Python Special Methods (Dunder Methods) - Context Management

This file provides a detailed explanation and implementation of Python's context management
special methods (__enter__, __exit__, __aenter__, __aexit__). These methods are part of
Python's protocol for implementing context managers, which are used with the 'with' statement.

We will cover:
1. Concept explanation
2. Detailed implementation
3. Practical examples
4. Best practices and error handling
5. Async context managers

All code follows PEP-8 standards and includes comprehensive explanations.
"""

# === 1. Concept Explanation ===

"""
Context Managers in Python:
- Context managers are objects that define the runtime context to be established when executing a 'with' statement.
- They are primarily used for resource management (e.g., files, database connections, locks).
- The 'with' statement ensures that resources are properly acquired and released, even if an error occurs.

Key Special Methods:
1. __enter__:
   - Called when entering the 'with' block.
   - It sets up the resource and returns the resource (or any object) to be used in the 'with' block.
   - Syntax: def __enter__(self)

2. __exit__:
   - Called when exiting the 'with' block, whether normally or due to an exception.
   - It handles cleanup (e.g., closing a file, releasing a lock).
   - Syntax: def __exit__(self, exc_type, exc_value, traceback)
   - Parameters:
     - exc_type: Type of the exception (None if no exception occurred)
     - exc_value: Exception instance (None if no exception occurred)
     - traceback: Traceback object (None if no exception occurred)
   - If __exit__ returns True, any exception is suppressed (not recommended unless intentional).

3. __aenter__ and __aexit__:
   - Asynchronous versions of __enter__ and __exit__ for use with 'async with' in asynchronous programming.
   - Used in async/await context for managing asynchronous resources (e.g., async database connections, network sockets).
   - Syntax: async def __aenter__(self) and async def __aexit__(self, exc_type, exc_value, traceback)
"""

# === 2. Detailed Implementation ===

class FileManager:
    """
    A synchronous context manager for file operations.
    Demonstrates __enter__ and __exit__.
    """
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        """
        Called when entering the 'with' block.
        - Opens the file and returns the file object.
        - The returned object is assigned to the 'as' variable in the 'with' statement.
        """
        print(f"Entering context: Opening file {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file  # This is what 'as f' will bind to

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Called when exiting the 'with' block.
        - Cleans up by closing the file.
        - Handles exceptions if any occurred in the 'with' block.
        """
        print(f"Exiting context: Closing file {self.filename}")
        if self.file:
            self.file.close()

        # If an exception occurred, print details
        if exc_type is not None:
            print(f"An exception occurred: {exc_type.__name__}: {exc_value}")
            # Returning False propagates the exception (default behavior)
            # Returning True would suppress the exception (not recommended unless intentional)
            return False


# === 3. Practical Examples ===

# Example 1: Using the FileManager context manager
def example_sync_file_manager():
    """
    Demonstrates the use of the FileManager context manager.
    Shows how resources are managed and exceptions are handled.
    """
    print("=== Example 1: Synchronous File Manager ===")
    try:
        with FileManager("example.txt", "w") as f:
            f.write("Hello, Context Manager!")
            # Simulate an error to demonstrate __exit__ handling
            raise ValueError("Simulated error inside 'with' block")
    except ValueError as e:
        print(f"Caught exception outside 'with' block: {e}")

    # Verify the file is closed by trying to write to it
    try:
        f.write("This should fail")
    except ValueError:
        print("File is closed, cannot write to it.")


# === 4. Asynchronous Context Managers ===

import asyncio

class AsyncDatabaseConnection:
    """
    An asynchronous context manager for database connections.
    Demonstrates __aenter__ and __aexit__.
    """
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None

    async def __aenter__(self):
        """
        Asynchronous version of __enter__.
        - Simulates establishing an async database connection.
        - Returns the connection object to be used in the 'async with' block.
        """
        print(f"Entering async context: Connecting to database {self.db_name}")
        # Simulate async connection setup
        await asyncio.sleep(1)  # Simulate network delay
        self.connection = f"Connection to {self.db_name}"
        return self.connection

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Asynchronous version of __exit__.
        - Cleans up by closing the async database connection.
        - Handles exceptions if any occurred in the 'async with' block.
        """
        print(f"Exiting async context: Disconnecting from database {self.db_name}")
        # Simulate async cleanup
        await asyncio.sleep(1)  # Simulate network delay
        self.connection = None

        if exc_type is not None:
            print(f"An async exception occurred: {exc_type.__name__}: {exc_value}")
            return False  # Propagate the exception


# Example 2: Using the AsyncDatabaseConnection context manager
async def example_async_db_connection():
    """
    Demonstrates the use of the AsyncDatabaseConnection context manager.
    Shows how async resources are managed and exceptions are handled.
    """
    print("=== Example 2: Asynchronous Database Connection ===")
    try:
        async with AsyncDatabaseConnection("my_database") as conn:
            print(f"Using connection: {conn}")
            # Simulate an error to demonstrate __aexit__ handling
            raise RuntimeError("Simulated error inside 'async with' block")
    except RuntimeError as e:
        print(f"Caught async exception outside 'async with' block: {e}")


# === 5. Best Practices and Error Handling ===

"""
Best Practices for Context Managers:
1. Always clean up resources in __exit__/__aexit__:
   - Ensure resources (files, connections, locks) are released, even if an exception occurs.
   - Use try-finally blocks inside __exit__ if additional cleanup logic is complex.

2. Handle exceptions appropriately:
   - In __exit__/__aexit__, return False to propagate exceptions (default behavior).
   - Only return True if you intentionally want to suppress exceptions (e.g., logging and ignoring).

3. Use context managers for resource management:
   - Prefer context managers over manual resource management to avoid resource leaks.

4. Keep __enter__/__aenter__ lightweight:
   - Do minimal setup in __enter__/__aenter__ to avoid blocking or delays.
   - Defer heavy initialization to the body of the 'with' block if possible.

5. Test edge cases:
   - Test context managers with exceptions, normal execution, and nested 'with' statements.
   - Ensure resources are always released, even in error cases.

Error Handling in Context Managers:
- When an exception occurs inside a 'with' block, __exit__/__aexit__ is still called.
- The exception details (exc_type, exc_value, traceback) are passed to __exit__/__aexit__.
- If __exit__/__aexit__ does not handle the exception (returns False), it is reraised after cleanup.
"""

# Example 3: Demonstrating best practices with error handling
class SafeResource:
    """
    A context manager demonstrating best practices and error handling.
    """
    def __init__(self, resource_name):
        self.resource_name = resource_name
        self.resource = None

    def __enter__(self):
        print(f"Acquiring resource: {self.resource_name}")
        self.resource = f"Active resource: {self.resource_name}"
        return self.resource

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Releasing resource: {self.resource_name}")
        self.resource = None
        if exc_type is not None:
            print(f"Handling exception during resource use: {exc_type.__name__}: {exc_value}")
            # Log the exception but do not suppress it
            return False


def example_safe_resource():
    """
    Demonstrates best practices and error handling with SafeResource.
    """
    print("=== Example 3: Safe Resource with Error Handling ===")
    try:
        with SafeResource("Database") as resource:
            print(f"Using resource: {resource}")
            raise ValueError("Error while using resource")
    except ValueError as e:
        print(f"Caught exception outside 'with' block: {e}")


# === 6. Running the Examples ===

if __name__ == "__main__":
    # Run synchronous examples
    example_sync_file_manager()
    print("\n")
    example_safe_resource()
    print("\n")

    # Run asynchronous example
    asyncio.run(example_async_db_connection())