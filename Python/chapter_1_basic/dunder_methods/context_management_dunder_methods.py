# context_management_dunder_methods.py
"""
Python Context Management Dunder Methods

This file explains the special methods (dunder methods) used for context management
in Python: __enter__, __exit__, __aenter__, and __aexit__.
"""

# =============================================================================
# Regular Context Managers (__enter__ and __exit__)
# =============================================================================

class FileHandler:
    """A context manager for file operations demonstrating __enter__ and __exit__."""
    
    def __init__(self, filename, mode='r'):
        """Initialize with filename and mode."""
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """
        The __enter__ method is called when entering a 'with' block.
        
        Purpose:
        - Acquires the resource (opens file in this case)
        - Returns an object that will be bound to the 'as' variable in the with statement
        - Executed when 'with' statement begins execution
        
        Return value:
        - The value returned will be assigned to the variable after 'as' in the with statement
        - Often returns self or the managed resource
        """
        self.file = open(self.filename, self.mode)
        return self.file  # This is what gets assigned to the 'as' variable
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The __exit__ method is called when exiting a 'with' block.
        
        Parameters:
        - exc_type: Exception type if an exception occurred, None otherwise
        - exc_val: Exception value if an exception occurred, None otherwise
        - exc_tb: Exception traceback if an exception occurred, None otherwise
        
        Purpose:
        - Releases the resource (closes file in this case)
        - Handles any exceptions that occurred in the with block
        - Always called, whether an exception occurred or not
        - Executed when 'with' block completes (normally or due to an exception)
        
        Return value:
        - Return True to suppress any exception that occurred in the with block
        - Return False or None to propagate any exception that occurred
        """
        if self.file:
            self.file.close()
        
        # Example of handling a specific exception
        if exc_type is FileNotFoundError:
            print(f"File {self.filename} not found!")
            return True  # Suppress the exception
        
        return False  # Propagate any other exceptions


# Example usage:
with FileHandler('example.txt', 'w') as f:
    f.write('Hello, World!')



# =============================================================================
# Asynchronous Context Managers (__aenter__ and __aexit__)
# =============================================================================

class AsyncDatabaseConnection:
    """
    An asynchronous context manager for database operations demonstrating
    __aenter__ and __aexit__ methods.
    """
    
    def __init__(self, connection_string):
        """Initialize with connection string."""
        self.connection_string = connection_string
        self.connection = None
    
    async def __aenter__(self):
        """
        The __aenter__ method is the asynchronous version of __enter__.
        
        Purpose:
        - Called when entering an 'async with' block
        - Must be a coroutine function (defined with 'async def')
        - Used to asynchronously acquire resources
        - Allows for non-blocking resource acquisition
        
        Return value:
        - The value returned will be assigned to the variable after 'as' in the async with statement
        - Often returns self or the managed resource
        """
        # Simulate async database connection
        print(f"Connecting to database: {self.connection_string}")
        # In a real scenario, we would use:
        # self.connection = await aiodblib.connect(self.connection_string)
        self.connection = {"connected": True, "id": "db-123"}
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        The __aexit__ method is the asynchronous version of __exit__.
        
        Parameters:
        - exc_type: Exception type if an exception occurred, None otherwise
        - exc_val: Exception value if an exception occurred, None otherwise
        - exc_tb: Exception traceback if an exception occurred, None otherwise
        
        Purpose:
        - Must be a coroutine function (defined with 'async def')
        - Called when exiting an 'async with' block
        - Used to asynchronously release resources
        - Handles any exceptions that occurred in the async with block
        - Always called, whether an exception occurred or not
        
        Return value:
        - Return True to suppress any exception that occurred in the async with block
        - Return False or None to propagate any exception that occurred
        """
        if self.connection:
            # In a real scenario: await self.connection.close()
            print(f"Disconnecting from database: {self.connection_string}")
            self.connection = None
        
        # Example of handling a specific exception
        if exc_type is ConnectionError:
            print("Database connection error occurred!")
            return True  # Suppress the exception
        
        return False  # Propagate any other exceptions


# Example usage:
# async def main():
#     async with AsyncDatabaseConnection("postgresql://user:pass@localhost/db") as conn:
#         # Perform database operations
#         print(f"Connected with ID: {conn['id']}")
#         await conn.execute("SELECT * FROM users")


# =============================================================================
# Advanced Context Manager Factory Example - Combining Both Patterns
# =============================================================================

class ContextManagerFactory:
    """
    A factory class that can create both synchronous and asynchronous
    context managers.
    """
    
    @staticmethod
    def create_logging_cm(log_name, async_mode=False):
        """
        Factory method to create a logging context manager.
        
        Args:
            log_name: Name for the log
            async_mode: Whether to create an async context manager
        
        Returns:
            A context manager class with appropriate dunder methods
        """
        if async_mode:
            # Create an async context manager class
            class AsyncLogManager:
                async def __aenter__(self):
                    print(f"[ASYNC] Starting log session: {log_name}")
                    return {"log_name": log_name, "async": True}
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if exc_type:
                        print(f"[ASYNC] Error in log session {log_name}: {exc_val}")
                    print(f"[ASYNC] Ending log session: {log_name}")
                    return False
            
            return AsyncLogManager()
        else:
            # Create a regular context manager class
            class LogManager:
                def __enter__(self):
                    print(f"Starting log session: {log_name}")
                    return {"log_name": log_name, "async": False}
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type:
                        print(f"Error in log session {log_name}: {exc_val}")
                    print(f"Ending log session: {log_name}")
                    return False
            
            return LogManager()


# Example usage:
# # Synchronous
# with ContextManagerFactory.create_logging_cm("app_log") as log:
#     print(f"Logging with: {log['log_name']}")
#
# # Asynchronous
# async def async_log_demo():
#     async with ContextManagerFactory.create_logging_cm("async_log", async_mode=True) as log:
#         print(f"Async logging with: {log['log_name']}")


# =============================================================================
# Key Points About Context Management Dunder Methods
# =============================================================================

"""
Key Facts About Context Management:

1. Regular Context Managers:
   - Use __enter__ and __exit__ methods
   - Used with the 'with' statement
   - __enter__ acquires resources and returns a value for the 'as' variable
   - __exit__ releases resources and handles exceptions

2. Async Context Managers:
   - Use __aenter__ and __aexit__ methods
   - Used with the 'async with' statement
   - Require Python 3.5+ and work with the asyncio module
   - Both methods must be coroutines (defined with 'async def')

3. Exception Handling:
   - __exit__ and __aexit__ receive exception information if an exception occurred
   - Return True to suppress exceptions, False/None to propagate them
   - This enables clean-up code to run even when exceptions occur

4. Implementation Details:
   - These methods are called by the Python interpreter, not directly by user code
   - The contextlib module provides utilities for creating context managers
   - The @contextmanager and @asynccontextmanager decorators offer simpler alternatives

5. Common Use Cases:
   - Resource management (files, connections, locks)
   - Transaction control (database transactions)
   - Temporary state changes
   - Timing and profiling
   - Error handling and logging
"""