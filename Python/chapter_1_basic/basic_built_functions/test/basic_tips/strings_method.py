# comprehensive_string_methods.py
"""
A comprehensive guide to Python's built-in string methods.
This file covers all aspects of each string method including syntax, usage examples,
edge cases, exceptions, and best practices.

"""


# ====================================================================================================
# str.capitalize()
# ====================================================================================================
def demo_capitalize():
    """
    str.capitalize(): Returns a copy of the string with its first character capitalized and the rest lowercased.
    
    Syntax:
        string.capitalize()
    
    Returns:
        A new string with the first character capitalized and the rest lowercased.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) as it creates a new string
    """
    # Basic example
    text = "hello world"
    capitalized = text.capitalize()
    print(f"Original: '{text}' → Capitalized: '{capitalized}'")  # Output: Original: 'hello world' → Capitalized: 'Hello world'
    
    # If the first character is not a letter, it remains unchanged
    text2 = "123abc"
    print(f"Original: '{text2}' → Capitalized: '{text2.capitalize()}'")  # Output: Original: '123abc' → Capitalized: '123abc'
    
    # All other letters are converted to lowercase
    text3 = "HELLO WORLD"
    print(f"Original: '{text3}' → Capitalized: '{text3.capitalize()}'")  # Output: Original: 'HELLO WORLD' → Capitalized: 'Hello world'
    
    # Empty string case
    empty = ""
    print(f"Empty string: '{empty.capitalize()}'")  # Output: Empty string: ''
    
    # Unicode support
    unicode_text = "éLÉPHANT"
    print(f"Original: '{unicode_text}' → Capitalized: '{unicode_text.capitalize()}'")  # Output: Original: 'éLÉPHANT' → Capitalized: 'Éléphant'
    
    # Common use case - correcting improperly capitalized titles
    incorrect_title = "tHE gREAT gATSBY"
    corrected = incorrect_title.lower().capitalize()
    print(f"Incorrect: '{incorrect_title}' → Corrected: '{corrected}'")  # Output: Incorrect: 'tHE gREAT gATSBY' → Corrected: 'The great gatsby'


# ====================================================================================================
# str.casefold()
# ====================================================================================================
def demo_casefold():
    """
    str.casefold(): Returns a casefolded copy of the string, suitable for case-insensitive comparisons.
    More aggressive than lower() - specifically designed for case-insensitive string comparisons.
    
    Syntax:
        string.casefold()
    
    Returns:
        A new casefolded string for caseless matching.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) as it creates a new string
    """
    # Basic example
    text = "Hello World"
    casefolded = text.casefold()
    print(f"Original: '{text}' → Casefolded: '{casefolded}'")  # Output: Original: 'Hello World' → Casefolded: 'hello world'
    
    # Comparison with lower()
    text2 = "Groß"  # German word with special character ß (eszett)
    print(f"lower(): '{text2.lower()}'")  # Output: 'groß'
    print(f"casefold(): '{text2.casefold()}'")  # Output: 'gross' - converts ß to ss
    
    # Case-insensitive comparison example
    word1 = "Straße"  # German for "street"
    word2 = "STRASSE"  # Uppercase version using 'ss' instead of 'ß'
    
    # Using lower() (fails)
    print(f"Using lower(): '{word1.lower()}' == '{word2.lower()}' → {word1.lower() == word2.lower()}")
    # Output: Using lower(): 'straße' == 'strasse' → False
    
    # Using casefold() (succeeds)
    print(f"Using casefold(): '{word1.casefold()}' == '{word2.casefold()}' → {word1.casefold() == word2.casefold()}")
    # Output: Using casefold(): 'strasse' == 'strasse' → True
    
    # Another international example
    greek_word1 = "ΜΆΙΚ"
    greek_word2 = "μάικ"
    print(f"Greek words casefold comparison: {greek_word1.casefold() == greek_word2.casefold()}")  # Output: True
    
    # Empty string
    empty = ""
    print(f"Empty string: '{empty.casefold()}'")  # Output: Empty string: ''


# ====================================================================================================
# str.center()
# ====================================================================================================
def demo_center():
    """
    str.center(width[, fillchar]): Returns a centered string of specified width.
    
    Syntax:
        string.center(width[, fillchar])
    
    Parameters:
        width (int): The total width of the resulting string
        fillchar (str, optional): The character to pad with. Defaults to space.
    
    Returns:
        A new string padded with the specified character to center the original string.
    
    Raises:
        TypeError: If width is not an integer, or fillchar is not a character.
    
    Time Complexity: O(n) where n is the width
    Space Complexity: O(n) for the new string
    """
    # Basic example
    text = "Python"
    centered = text.center(20)
    print(f"Original: '{text}' → Centered: '{centered}'")  # Output: Original: 'Python' → Centered: '       Python       '
    
    # Using a custom fill character
    centered_with_star = text.center(20, '*')
    print(f"Centered with '*': '{centered_with_star}'")  # Output: Centered with '*': '*******Python*******'
    
    # When width is less than or equal to the length of the string
    print(f"Width equal to length: '{text.center(6)}'")  # Output: Width equal to length: 'Python'
    print(f"Width less than length: '{text.center(3)}'")  # Output: Width less than length: 'Python'
    
    # Uneven padding (one side gets one more character)
    text2 = "Hi"
    print(f"Uneven padding: '{text2.center(5, '-')}'")  # Output: Uneven padding: '--Hi-'
    
    # Empty string
    empty = ""
    print(f"Empty string: '{empty.center(5, "*")}'")  # Output: Empty string: '*****'
    
    # Unicode/emoji support
    emoji = "🐍"
    print(f"Centered emoji: '{emoji.center(5, ".")}'")  # Output: Centered emoji: '..🐍..'
    
    # Exception cases
    try:
        # TypeError: center() argument 1 must be int, not str
        text.center("20")
    except TypeError as e:
        print(f"Error with non-integer width: {e}")
    
    try:
        # TypeError: center() argument 2 must be str of length 1, not str
        text.center(20, "**")
    except TypeError as e:
        print(f"Error with multi-character fillchar: {e}")


# ====================================================================================================
# str.count()
# ====================================================================================================
def demo_count():
    """
    str.count(sub[, start[, end]]): Returns the number of non-overlapping occurrences of substring.
    
    Syntax:
        string.count(substring[, start[, end]])
    
    Parameters:
        sub (str): The substring to count
        start (int, optional): The starting index. Defaults to 0.
        end (int, optional): The ending index. Defaults to the end of the string.
    
    Returns:
        int: The number of non-overlapping occurrences of the substring.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1) as it only returns a count
    """
    # Basic example
    text = "Mississippi"
    count_s = text.count('s')
    print(f"Count of 's' in '{text}': {count_s}")  # Output: Count of 's' in 'Mississippi': 4
    
    # Counting a substring
    count_ss = text.count('ss')
    print(f"Count of 'ss' in '{text}': {count_ss}")  # Output: Count of 'ss' in 'Mississippi': 2
    
    # Using start and end parameters
    count_s_partial = text.count('s', 3, 8)
    print(f"Count of 's' from index 3 to 8 in '{text}': {count_s_partial}")  # Output: Count of 's' in 'Mississippi' from index 3 to 8: 2
    
    # Non-overlapping occurrences
    text2 = "abababa"
    print(f"Count of 'aba' in '{text2}': {text2.count('aba')}")  # Output: Count of 'aba' in 'abababa': 2 (non-overlapping)
    
    # Case sensitivity
    text3 = "Hello hello HELLO"
    print(f"Count of 'hello' (case-sensitive): {text3.count('hello')}")  # Output: Count of 'hello' (case-sensitive): 1
    print(f"Count of 'hello' (case-insensitive): {text3.lower().count('hello')}")  # Output: Count of 'hello' (case-insensitive): 3
    
    # Empty string or substring
    print(f"Count of '' in 'abc': {'abc'.count('')}")  # Output: Count of '' in 'abc': 4 (empty string is found at each position, including before the first and after the last character)
    print(f"Count of 'xyz' in 'abc': {'abc'.count('xyz')}")  # Output: Count of 'xyz' in 'abc': 0
    
    # Out of range indices (no error, just returns 0)
    print(f"Count with out of range indices: {'abc'.count('a', 10, 20)}")  # Output: Count with out of range indices: 0
    
    # Negative indices (counts from the end)
    text4 = "banana"
    print(f"Count of 'a' from -4 to -1: {text4.count('a', -4, -1)}")  # Output: Count of 'a' from -4 to -1: 1


# ====================================================================================================
# str.encode()
# ====================================================================================================
def demo_encode():
    """
    str.encode([encoding[, errors]]): Returns an encoded version of the string as a bytes object.
    
    Syntax:
        string.encode(encoding='utf-8', errors='strict')
    
    Parameters:
        encoding (str, optional): The encoding to use. Defaults to 'utf-8'.
        errors (str, optional): The error handling scheme. Options:
            'strict': Raises UnicodeError on failure (default)
            'ignore': Ignores characters that cannot be encoded
            'replace': Replaces characters with a replacement marker (e.g., ?)
            'xmlcharrefreplace': Replaces with XML character reference
            'backslashreplace': Replaces with a backslashed escape sequence
            'namereplace': Replaces with \N{...} escape sequences
    
    Returns:
        bytes: An encoded version of the string
    
    Raises:
        UnicodeEncodeError: When a character cannot be encoded in the specified encoding (in strict mode)
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) for the bytes object
    """
    # Basic example with UTF-8 (default)
    text = "Hello, world!"
    encoded = text.encode()  # Default is utf-8
    print(f"Original: '{text}' → UTF-8 encoded: {encoded}")  # Output: Original: 'Hello, world!' → UTF-8 encoded: b'Hello, world!'
    
    # Unicode characters with different encodings
    unicode_text = "こんにちは世界"  # "Hello world" in Japanese
    
    # UTF-8 encoding (variable length, efficient for ASCII)
    utf8_encoded = unicode_text.encode('utf-8')
    print(f"UTF-8 encoded: {utf8_encoded}")  # Output: UTF-8 encoded: b'\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe4\xb8\x96\xe7\x95\x8c'
    
    # UTF-16 encoding (uses 2 or 4 bytes per character)
    utf16_encoded = unicode_text.encode('utf-16')
    print(f"UTF-16 encoded: {utf16_encoded}")  # Output has BOM (Byte Order Mark) at start
    
    # ASCII encoding (only supports ASCII characters)
    try:
        ascii_encoded = unicode_text.encode('ascii')
    except UnicodeEncodeError as e:
        print(f"ASCII encoding error: {e}")
    
    # Error handling methods
    print("\nError handling methods:")
    
    # 'strict' (default) - raises an error for unencodable characters
    try:
        unicode_text.encode('ascii', 'strict')
    except UnicodeEncodeError as e:
        print(f"'strict': {e}")
    
    # 'ignore' - silently drops unencodable characters
    ascii_ignore = unicode_text.encode('ascii', 'ignore')
    print(f"'ignore': {ascii_ignore}")  # Output: b'' (all characters ignored)
    
    # 'replace' - replaces unencodable characters with a replacement character (?)
    ascii_replace = unicode_text.encode('ascii', 'replace')
    print(f"'replace': {ascii_replace}")  # Output: b'??????' (question marks)
    
    # 'xmlcharrefreplace' - replaces with XML character references
    ascii_xml = unicode_text.encode('ascii', 'xmlcharrefreplace')
    print(f"'xmlcharrefreplace': {ascii_xml}")  # Output: XML character references
    
    # 'backslashreplace' - replaces with backslashed escape sequences
    ascii_backslash = unicode_text.encode('ascii', 'backslashreplace')
    print(f"'backslashreplace': {ascii_backslash}")  # Output: Backslashed escape sequences
    
    # 'namereplace' - replaces with \N{...} escape sequences
    ascii_name = unicode_text.encode('ascii', 'namereplace')
    print(f"'namereplace': {ascii_name}")  # Output: Named escape sequences
    
    # Encoding an empty string
    empty = ""
    print(f"Empty string encoded: {empty.encode()}")  # Output: Empty string encoded: b''


# ====================================================================================================
# str.endswith()
# ====================================================================================================
def demo_endswith():
    """
    str.endswith(suffix[, start[, end]]): Returns True if the string ends with the specified suffix.
    
    Syntax:
        string.endswith(suffix[, start[, end]])
    
    Parameters:
        suffix (str or tuple of str): The suffix(es) to check
        start (int, optional): The starting index. Defaults to 0.
        end (int, optional): The ending index. Defaults to the end of the string.
    
    Returns:
        bool: True if the string ends with the specified suffix, False otherwise
    
    Time Complexity: O(n) where n is the length of the suffix
    Space Complexity: O(1)
    """
    # Basic example
    filename = "document.pdf"
    is_pdf = filename.endswith('.pdf')
    print(f"'{filename}' ends with '.pdf': {is_pdf}")  # Output: 'document.pdf' ends with '.pdf': True
    
    # Case sensitivity
    filename2 = "image.PNG"
    print(f"'{filename2}' ends with '.png': {filename2.endswith('.png')}")  # Output: 'image.PNG' ends with '.png': False
    print(f"'{filename2}' ends with '.png' (case-insensitive): {filename2.lower().endswith('.png')}")  # Output: True
    
    # Using start and end parameters
    text = "Python programming is fun"
    print(f"'{text}' from index 0 to 6 ends with 'on': {text.endswith('on', 0, 6)}")  # Output: True ('Python' ends with 'on')
    
    # Multiple suffixes using a tuple
    file_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    image_file = "vacation.jpg"
    print(f"'{image_file}' is an image: {image_file.endswith(file_extensions)}")  # Output: True
    
    # Empty string checks
    print(f"'' ends with '': {''.endswith('')}")  # Output: True (empty string is a suffix of any string)
    print(f"'abc' ends with '': {'abc'.endswith('')}")  # Output: True (empty string is a suffix of any string)
    print(f"'' ends with 'x': {''.endswith('x')}")  # Output: False (non-empty string is not a suffix of empty string)
    
    # Negative indices
    text2 = "Hello, world!"
    print(f"'{text2}' ends with 'world' when using negative indices: {text2.endswith('world', -13, -1)}")  # Output: True
    
    # Practical example: checking file types
    files = ["report.docx", "image.jpg", "data.csv", "presentation.pptx", "notes.txt"]
    document_extensions = ('.doc', '.docx', '.pdf', '.txt')
    
    document_files = [file for file in files if file.endswith(document_extensions)]
    print(f"Document files: {document_files}")  # Output: Document files: ['report.docx', 'notes.txt']


# ====================================================================================================
# str.expandtabs()
# ====================================================================================================
def demo_expandtabs():
    """
    str.expandtabs([tabsize]): Returns a copy of the string with all tab characters replaced by spaces.
    
    Syntax:
        string.expandtabs(tabsize=8)
    
    Parameters:
        tabsize (int, optional): The number of spaces to replace each tab with. Defaults to 8.
    
    Returns:
        str: A new string with tabs expanded to spaces
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) for the new string
    """
    # Basic example
    text_with_tabs = "Name\tAge\tCity"
    expanded = text_with_tabs.expandtabs()  # Default tabsize is 8
    print(f"Original: '{text_with_tabs}'")
    print(f"Expanded (default): '{expanded}'")
    
    # Different tabsize values
    print(f"Expanded (tabsize=4): '{text_with_tabs.expandtabs(4)}'")
    print(f"Expanded (tabsize=16): '{text_with_tabs.expandtabs(16)}'")
    
    # Tab positions are calculated to multiples of tabsize
    code_with_tabs = "\tdef hello():\n\t\tprint('Hello, world!')"
    print("\nCode with tabs:")
    print(code_with_tabs)
    print("\nExpanded tabs (tabsize=4):")
    print(code_with_tabs.expandtabs(4))
    
    # Tab after characters
    text2 = "Col1\tCol2\tCol3"
    text3 = "Column1\tColumn2\tColumn3"
    
    print("\nShowing tab alignment behavior:")
    print(f"Original 1: '{text2}'")
    print(f"Expanded 1: '{text2.expandtabs(8)}'")
    print(f"Original 2: '{text3}'")
    print(f"Expanded 2: '{text3.expandtabs(8)}'")
    
    # Complex example showing how tabstops work
    tabbed_text = "0123\t456\t789\t0"
    print("\nTab behavior example:")
    print(f"Original: '{tabbed_text}'")
    print(f"Expanded (tabsize=4): '{tabbed_text.expandtabs(4)}'")
    print(f"Expanded (tabsize=8): '{tabbed_text.expandtabs(8)}'")
    
    # The first tab expands to the next multiple of tabsize
    # In tabsize=4: 0123 already takes 4 spaces, so tab goes to position 8
    # In tabsize=8: 0123 takes 4 spaces, so tab goes to position 8
    
    # With tabsize=0, tabs are simply removed
    print(f"Expanded (tabsize=0): '{tabbed_text.expandtabs(0)}'")
    
    # Empty string
    empty = ""
    print(f"Empty string: '{empty.expandtabs()}'")  # Output: Empty string: ''
    
    # Negative tabsize is treated as 1
    print(f"Negative tabsize: '{tabbed_text.expandtabs(-4)}'")  # Treats as tabsize=1


# ====================================================================================================
# str.find()
# ====================================================================================================
def demo_find():
    """
    str.find(sub[, start[, end]]): Returns the lowest index where substring is found.
    
    Syntax:
        string.find(substring[, start[, end]])
    
    Parameters:
        sub (str): The substring to find
        start (int, optional): The starting index. Defaults to 0.
        end (int, optional): The ending index. Defaults to the end of the string.
    
    Returns:
        int: The lowest index where substring is found, or -1 if not found
    
    Time Complexity: O(n*m) where n is the length of the string and m is the length of the substring
    Space Complexity: O(1)
    """
    # Basic example
    text = "Python is a programming language. Python is easy to learn."
    index = text.find("Python")
    print(f"First occurrence of 'Python' in '{text}': index {index}")  # Output: First occurrence of 'Python': index 0
    
    # Find with start parameter
    index2 = text.find("Python", 10)
    print(f"First occurrence after index 10: index {index2}")  # Output: First occurrence after index 10: index 35
    
    # Find with start and end parameters
    index3 = text.find("language", 0, 30)
    print(f"'language' between indices 0 and 30: index {index3}")  # Output: 'language' between indices 0 and 30: index 23
    
    # Substring not found
    not_found = text.find("Java")
    print(f"'Java' in text: index {not_found}")  # Output: 'Java' in text: index -1
    
    # Case sensitivity
    case_sensitive = text.find("python")
    print(f"'python' (lowercase) in text: index {case_sensitive}")  # Output: 'python' (lowercase) in text: index -1
    case_insensitive = text.lower().find("python")
    print(f"'python' (case-insensitive): index {case_insensitive}")  # Output: 'python' (case-insensitive): index 0
    
    # Finding empty string
    empty_string = text.find("")
    print(f"Empty string in text: index {empty_string}")  # Output: Empty string in text: index 0 (empty string is found at every position)
    
    # Negative indices
    negative_indices = text.find("is", -30, -5)
    print(f"'is' with negative indices: index {negative_indices}")  # Output depends on the text
    
    # Common use case: checking if substring exists
    if text.find("easy") != -1:
        print("The text contains 'easy'")
    else:
        print("The text does not contain 'easy'")
    
    # Practical example: Extracting a substring between two markers
    html = "<title>Python Documentation</title>"
    start_pos = html.find("<title>") + len("<title>")
    end_pos = html.find("</title>")
    
    if start_pos != -1 and end_pos != -1:
        title = html[start_pos:end_pos]
        print(f"Extracted title: '{title}'")  # Output: Extracted title: 'Python Documentation'
    
    # Difference from index(): find() returns -1 instead of raising ValueError
    try:
        text.index("nonexistent")
    except ValueError as e:
        print(f"text.index() raised: {e}")
    
    print(f"text.find() returned: {text.find('nonexistent')}")  # Output: text.find() returned: -1


# ====================================================================================================
# str.format()
# ====================================================================================================
def demo_format():
    """
    str.format(*args, **kwargs): Formats the string using the specified values.
    
    Syntax:
        string.format(*args, **kwargs)
    
    Parameters:
        *args: Positional arguments to be formatted into the string
        **kwargs: Keyword arguments to be formatted into the string
    
    Returns:
        str: A formatted string
    
    Raises:
        KeyError: When a named placeholder is not provided
        IndexError: When an index for a positional argument is out of range
        ValueError: When an invalid format specification is used
    
    Time Complexity: O(n) where n is the length of the final formatted string
    Space Complexity: O(n) for the new string
    """
    # Basic positional formatting
    template1 = "Hello, {}. You are {} years old."
    formatted1 = template1.format("Alice", 30)
    print(f"Positional formatting: '{formatted1}'")  # Output: Positional formatting: 'Hello, Alice. You are 30 years old.'
    
    # Index-based positional formatting
    template2 = "The {1} in the {0}."
    formatted2 = template2.format("hat", "cat")
    print(f"Index-based formatting: '{formatted2}'")  # Output: Index-based formatting: 'The cat in the hat.'
    
    # Reusing positional arguments
    template3 = "{0} {1}. {0} {1}. {0} {1}!"
    formatted3 = template3.format("Go", "team")
    print(f"Reusing positional args: '{formatted3}'")  # Output: Reusing positional args: 'Go team. Go team. Go team!'
    
    # Keyword arguments
    template4 = "Hello, {name}. You are {age} years old."
    formatted4 = template4.format(name="Bob", age=25)
    print(f"Keyword formatting: '{formatted4}'")  # Output: Keyword formatting: 'Hello, Bob. You are 25 years old.'
    
    # Accessing object attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    
    person = Person("Charlie", 22)
    template5 = "Person: {p.name}, Age: {p.age}"
    formatted5 = template5.format(p=person)
    print(f"Attribute access: '{formatted5}'")  # Output: Attribute access: 'Person: Charlie, Age: 22'
    
    # Accessing dictionary items
    data = {"name": "David", "age": 35}
    template6 = "Person: {d[name]}, Age: {d[age]}"
    formatted6 = template6.format(d=data)
    print(f"Dictionary access: '{formatted6}'")  # Output: Dictionary access: 'Person: David, Age: 35'
    
    # Formatting options
    # Width and alignment
    print("\nWidth and alignment:")
    print("'{:10}'.format('left')      -> '{}'".format('{:10}'.format('left')))      # Right-aligned in 10 spaces
    print("'{:>10}'.format('right')    -> '{}'".format('{:>10}'.format('right')))    # Right-aligned in 10 spaces
    print("'{:<10}'.format('left')     -> '{}'".format('{:<10}'.format('left')))     # Left-aligned in 10 spaces
    print("'{:^10}'.format('center')   -> '{}'".format('{:^10}'.format('center')))   # Centered in 10 spaces
    
    # Fill character
    print("\nFill character:")
    print("'{:*>10}'.format('right')   -> '{}'".format('{:*>10}'.format('right')))   # Right-aligned with * padding
    print("'{:*<10}'.format('left')    -> '{}'".format('{:*<10}'.format('left')))    # Left-aligned with * padding
    print("'{:*^10}'.format('center')  -> '{}'".format('{:*^10}'.format('center')))  # Centered with * padding
    
    # Number formatting
    print("\nNumber formatting:")
    print("'{:d}'.format(42)          -> '{}'".format('{:d}'.format(42)))          # Integer
    print("'{:f}'.format(3.14159)     -> '{}'".format('{:f}'.format(3.14159)))     # Float (default precision 6)
    print("'{:.2f}'.format(3.14159)   -> '{}'".format('{:.2f}'.format(3.14159)))   # Float with 2 decimal places
    print("'{:,}'.format(1234567890)  -> '{}'".format('{:,}'.format(1234567890)))  # Number with thousand separators
    print("'{:.2%}'.format(0.25)      -> '{}'".format('{:.2%}'.format(0.25)))      # Percentage format
    
    # Binary, octal, hexadecimal
    print("\nBase conversion:")
    print("'{:b}'.format(42)          -> '{}'".format('{:b}'.format(42)))          # Binary
    print("'{:o}'.format(42)          -> '{}'".format('{:o}'.format(42)))          # Octal
    print("'{:x}'.format(42)          -> '{}'".format('{:x}'.format(42)))          # Hexadecimal (lowercase)
    print("'{:X}'.format(42)          -> '{}'".format('{:X}'.format(42)))          # Hexadecimal (uppercase)
    print("'{:#b}'.format(42)         -> '{}'".format('{:#b}'.format(42)))         # Binary with 0b prefix
    print("'{:#o}'.format(42)         -> '{}'".format('{:#o}'.format(42)))         # Octal with 0o prefix
    print("'{:#x}'.format(42)         -> '{}'".format('{:#x}'.format(42)))         # Hex with 0x prefix
    
    # Date formatting
    import datetime
    now = datetime.datetime(2025, 3, 14, 15, 9, 26)
    print("\nDate formatting:")
    print("datetime: {}".format(now))
    print("'{:%Y-%m-%d %H:%M:%S}'.format(now) -> '{}'".format('{:%Y-%m-%d %H:%M:%S}'.format(now)))
    
    # Escaping braces
    print("\nEscaping braces:")
    print("'{{}} {{}}'.format()       -> '{}'".format('{{}} {{}}'.format()))       # Escaped braces
    
    # Error cases
    print("\nCommon errors:")
    try:
        "{missing}".format()  # KeyError: 'missing'
    except KeyError as e:
        print(f"KeyError: {e}")
    
    try:
        "{0} {1} {2}".format("too", "few")  # IndexError
    except IndexError as e:
        print(f"IndexError: {e}")
    
    try:
        "{:z}".format(123)  # ValueError: Unknown format code 'z'
    except ValueError as e:
        print(f"ValueError: {e}")


# ====================================================================================================
# str.format_map()
# ====================================================================================================
def demo_format_map():
    """
    str.format_map(mapping): Similar to str.format(**mapping), but uses the mapping directly without copying.
    
    Syntax:
        string.format_map(mapping)
    
    Parameters:
        mapping: A dictionary-like object that maps keys to values
    
    Returns:
        str: A formatted string
    
    Raises:
        KeyError: When a named placeholder is not in the mapping
    
    Time Complexity: O(n) where n is the length of the final formatted string
    Space Complexity: O(n) for the new string
    """
    # Basic example
    data = {"name": "Alice", "age": 30}
    template = "Name: {name}, Age: {age}"
    formatted = template.format_map(data)
    print(f"Basic format_map: '{formatted}'")  # Output: Basic format_map: 'Name: Alice, Age: 30'
    
    # Comparison with format(**data)
    formatted2 = template.format(**data)
    print(f"Using format(**data): '{formatted2}'")  # Output: Using format(**data): 'Name: Alice, Age: 30'
    
    # format_map advantage: works with custom dict-like classes that might not support **unpacking
    class DefaultDict(dict):
        def __missing__(self, key):
            return f"[{key} NOT FOUND]"
    
    incomplete_data = DefaultDict({"name": "Bob"})  # Missing 'age'
    formatted3 = template.format_map(incomplete_data)
    print(f"With missing key handler: '{formatted3}'")  # Output: With missing key handler: 'Name: Bob, Age: [age NOT FOUND]'
    
    # format_map with object attributes
    class Person:
        def __init__(self, name, age=None):
            self.name = name
            self.age = age
    
    class PersonDict(dict):
        def __missing__(self, key):
            return None
    
    person = Person("Charlie", 25)
    person_dict = PersonDict(vars(person))  # Convert object attributes to dict
    
    template2 = "Person: {name}, Age: {age}"
    formatted4 = template2.format_map(person_dict)
    print(f"Object attributes: '{formatted4}'")  # Output: Object attributes: 'Person: Charlie, Age: 25'
    
    # Handling missing values more gracefully
    template3 = "Person: {name}, Age: {age or 'N/A'}"
    person2 = Person("David")  # No age provided
    person_dict2 = PersonDict(vars(person2))
    formatted5 = template3.format_map(person_dict2)
    print(f"Handling None values: '{formatted5}'")  # Output: Handling None values: 'Person: David, Age: N/A'
    
    # Practical example: string substitution in a template
    template4 = """
    Dear {customer},
    
    Thank you for your order of {product}.
    Your order #{order_number} will be shipped on {ship_date}.
    
    Regards,
    {company}
    """
    
    order_data = {
        "customer": "Eve",
        "product": "Python Programming Book",
        "order_number": "ORD-12345",
        "ship_date": "2025-03-20",
        "company": "PythonBooks Inc."
    }
    
    formatted6 = template4.format_map(order_data)
    print("\nTemplate substitution:")
    print(formatted6)
    
    # Error case: missing key
    try:
        "Hello, {missing}".format_map({})  # KeyError: 'missing'
    except KeyError as e:
        print(f"KeyError: {e}")


# ====================================================================================================
# str.index()
# ====================================================================================================
def demo_index():
    """
    str.index(sub[, start[, end]]): Like find(), but raises ValueError when the substring is not found.
    
    Syntax:
        string.index(substring[, start[, end]])
    
    Parameters:
        sub (str): The substring to find
        start (int, optional): The starting index. Defaults to 0.
        end (int, optional): The ending index. Defaults to the end of the string.
    
    Returns:
        int: The lowest index where substring is found
    
    Raises:
        ValueError: When the substring is not found
    
    Time Complexity: O(n*m) where n is the length of the string and m is the length of the substring
    Space Complexity: O(1)
    """
    # Basic example
    text = "Python is amazing"
    index = text.index("is")
    print(f"Index of 'is' in '{text}': {index}")  # Output: Index of 'is' in 'Python is amazing': 7
    
    # Using start and end parameters
    index2 = text.index("a", 10)
    print(f"Index of 'a' after position 10: {index2}")  # Output: Index of 'a' after position 10: 13
    
    # Comparison with find()
    print("\nComparison with find():")
    print(f"find('a'): {text.find('a')}")  # Output: find('a'): 10
    print(f"index('a'): {text.index('a')}")  # Output: index('a'): 10
    
    # Handling substring not found
    print("\nHandling substring not found:")
    
    # Using find() (returns -1)
    not_found = text.find("xyz")
    print(f"find('xyz'): {not_found}")  # Output: find('xyz'): -1
    
    # Using index() (raises ValueError)
    try:
        text.index("xyz")
    except ValueError as e:
        print(f"index('xyz') raised: {e}")  # Output: index('xyz') raised: substring not found
    
    # Recommended pattern for checking if substring exists
    try:
        position = text.index("amazing")
        print(f"'amazing' found at index {position}")
    except ValueError:
        print("'amazing' not found in the string")
    
    # Empty string as a substring
    empty_index = text.index("")
    print(f"Index of empty string: {empty_index}")  # Output: Index of empty string: 0
    
    # Case sensitivity
    try:
        text.index("PYTHON")  # Case-sensitive
    except ValueError as e:
        print(f"Case sensitivity: {e}")  # Output: Case sensitivity: substring not found
    
    # Using negative indices
    index3 = text.index("amazing", -10)
    print(f"Index of 'amazing' with negative start: {index3}")  # Output depends on string length
    
    # Practical example: Extracting a substring between markers
    html = "<div>Content</div>"
    try:
        start_tag = html.index("<div>") + len("<div>")
        end_tag = html.index("</div>")
        content = html[start_tag:end_tag]
        print(f"Extracted content: '{content}'")  # Output: Extracted content: 'Content'
    except ValueError as e:
        print(f"Error: {e}")


# ====================================================================================================
# str.isalnum()
# ====================================================================================================
def demo_isalnum():
    """
    str.isalnum(): Returns True if all characters in the string are alphanumeric (letters or numbers).
    
    Syntax:
        string.isalnum()
    
    Returns:
        bool: True if all characters are alphanumeric and there is at least one character, False otherwise
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'abc123'.isalnum(): {'abc123'.isalnum()}")  # Output: 'abc123'.isalnum(): True
    print(f"'abc 123'.isalnum(): {'abc 123'.isalnum()}")  # Output: 'abc 123'.isalnum(): False (space is not alphanumeric)
    print(f"'abc-123'.isalnum(): {'abc-123'.isalnum()}")  # Output: 'abc-123'.isalnum(): False (hyphen is not alphanumeric)
    
    # Empty string
    print(f"''.isalnum(): {''.isalnum()}")  # Output: ''.isalnum(): False (empty string)
    
    # Unicode support
    print(f"'éçñ123'.isalnum(): {'éçñ123'.isalnum()}")  # Output: 'éçñ123'.isalnum(): True (Unicode letters are alphanumeric)
    
    # Only letters
    print(f"'HelloWorld'.isalnum(): {'HelloWorld'.isalnum()}")  # Output: 'HelloWorld'.isalnum(): True
    
    # Only numbers
    print(f"'12345'.isalnum(): {'12345'.isalnum()}")  # Output: '12345'.isalnum(): True
    
    # Symbols and punctuation
    print(f"'Hello!'.isalnum(): {'Hello!'.isalnum()}")  # Output: 'Hello!'.isalnum(): False (! is not alphanumeric)
    print(f"'123.45'.isalnum(): {'123.45'.isalnum()}")  # Output: '123.45'.isalnum(): False (. is not alphanumeric)
    
    # Practical examples
    # 1. Checking if a string can be used as a username (alphanumeric only)
    username = "user123"
    if username.isalnum():
        print(f"'{username}' is a valid username")
    else:
        print(f"'{username}' contains invalid characters")
    
    # 2. Filtering out non-alphanumeric strings from a list
    strings = ["abc", "123", "abc123", "hello world", "test@example", "12-34"]
    alphanumeric_strings = [s for s in strings if s.isalnum()]
    print(f"Alphanumeric strings: {alphanumeric_strings}")  # Output: Alphanumeric strings: ['abc', '123', 'abc123']
    
    # 3. Removing non-alphanumeric characters from a string
    text = "Hello, World! 123"
    alphanumeric_only = ''.join(char for char in text if char.isalnum())
    print(f"Original: '{text}' → Alphanumeric only: '{alphanumeric_only}'")  # Output: Original: 'Hello, World! 123' → Alphanumeric only: 'HelloWorld123'


# ====================================================================================================
# str.isalpha()
# ====================================================================================================
def demo_isalpha():
    """
    str.isalpha(): Returns True if all characters in the string are alphabetic (letters).
    
    Syntax:
        string.isalpha()
    
    Returns:
        bool: True if all characters are alphabetic and there is at least one character, False otherwise
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'abc'.isalpha(): {'abc'.isalpha()}")  # Output: 'abc'.isalpha(): True
    print(f"'abc123'.isalpha(): {'abc123'.isalpha()}")  # Output: 'abc123'.isalpha(): False (contains numbers)
    print(f"'Hello World'.isalpha(): {'Hello World'.isalpha()}")  # Output: 'Hello World'.isalpha(): False (contains space)
    
    # Empty string
    print(f"''.isalpha(): {''.isalpha()}")  # Output: ''.isalpha(): False (empty string)
    
    # Unicode support
    print(f"'éçñüö'.isalpha(): {'éçñüö'.isalpha()}")  # Output: 'éçñüö'.isalpha(): True (Unicode letters are alphabetic)
    print(f"'こんにちは'.isalpha(): {'こんにちは'.isalpha()}")  # Output: 'こんにちは'.isalpha(): True (Japanese characters are alphabetic)
    
    # Numbers and symbols
    print(f"'123'.isalpha(): {'123'.isalpha()}")  # Output: '123'.isalpha(): False (only numbers)
    print(f"'Hello!'.isalpha(): {'Hello!'.isalpha()}")  # Output: 'Hello!'.isalpha(): False (contains punctuation)
    
    # Whitespace
    print(f"' '.isalpha(): {' '.isalpha()}")  # Output: ' '.isalpha(): False (whitespace is not alphabetic)
    
    # Practical examples
    # 1. Checking if a string contains only letters (e.g., for names)
    name = "John"
    if name.isalpha():
        print(f"'{name}' contains only letters")
    else:
        print(f"'{name}' contains non-letter characters")
    
    # 2. Validating that a word contains only letters
    def is_valid_word(word):
        return word.isalpha()
    
    words = ["apple", "orange2", "banana!", "grape"]
    valid_words = [word for word in words if is_valid_word(word)]
    print(f"Valid words: {valid_words}")  # Output: Valid words: ['apple', 'grape']
    
    # 3. Checking words in a sentence
    sentence = "The quick brown fox jumps over the lazy dog"
    words = sentence.split()
    all_alpha_words = [word for word in words if word.isalpha()]
    print(f"Words with only letters: {all_alpha_words}")  # Output: all words in the sentence
    
    # 4. Language detection (simplified example)
    def detect_latin_based(text):
        """Simplified check if text appears to be Latin-based"""
        # Remove spaces and check if remaining characters are in Latin alphabet
        no_spaces = text.replace(" ", "")
        return no_spaces.isalpha() and all(ord(c) < 1000 for c in no_spaces)
    
    print(f"'Hello world' is Latin-based: {detect_latin_based('Hello world')}")  # Output: True
    print(f"'こんにちは世界' is Latin-based: {detect_latin_based('こんにちは世界')}")  # Output: False


# ====================================================================================================
# str.isascii()
# ====================================================================================================
def demo_isascii():
    """
    str.isascii(): Returns True if all characters in the string are ASCII.
    
    Syntax:
        string.isascii()
    
    Returns:
        bool: True if all characters are in the ASCII character set, False otherwise
    
    Note:
        This method was added in Python 3.7
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'hello'.isascii(): {'hello'.isascii()}")  # Output: 'hello'.isascii(): True
    print(f"'hello123!@#'.isascii(): {'hello123!@#'.isascii()}")  # Output: 'hello123!@#'.isascii(): True (ASCII includes letters, numbers, and common symbols)
    
    # Non-ASCII characters
    print(f"'héllo'.isascii(): {'héllo'.isascii()}")  # Output: 'héllo'.isascii(): False (é is not ASCII)
    print(f"'こんにちは'.isascii(): {'こんにちは'.isascii()}")  # Output: 'こんにちは'.isascii(): False (Japanese characters are not ASCII)
    print(f"'😀'.isascii(): {'😀'.isascii()}")  # Output: '😀'.isascii(): False (emoji are not ASCII)
    
    # Empty string
    print(f"''.isascii(): {''.isascii()}")  # Output: ''.isascii(): True (empty string has no non-ASCII characters)
    
    # ASCII control characters (0-31 and 127)
    print(f"'\\n\\t\\r'.isascii(): {'\n\t\r'.isascii()}")  # Output: '\n\t\r'.isascii(): True (control characters are ASCII)
    
    # Practical examples
    # 1. Checking if a file can be safely opened in ASCII mode
    def is_ascii_safe(text):
        return text.isascii()
    
    text1 = "Hello, world!"
    text2 = "Héllo, wörld!"
    
    print(f"'{text1}' can be safely saved as ASCII: {is_ascii_safe(text1)}")  # Output: True
    print(f"'{text2}' can be safely saved as ASCII: {is_ascii_safe(text2)}")  # Output: False
    
    # 2. Filtering out non-ASCII characters from a string
    def remove_non_ascii(text):
        return ''.join(char for char in text if ord(char) < 128)
    
    mixed_text = "Café 123"
    ascii_only = remove_non_ascii(mixed_text)
    print(f"Original: '{mixed_text}' → ASCII only: '{ascii_only}'")  # Output: Original: 'Café 123' → ASCII only: 'Caf 123'
    
    # 3. Checking if a username contains only ASCII characters
    def is_valid_username(username):
        return username.isascii() and username.isalnum()
    
    username1 = "user123"
    username2 = "usér123"
    
    print(f"'{username1}' is a valid ASCII username: {is_valid_username(username1)}")  # Output: True
    print(f"'{username2}' is a valid ASCII username: {is_valid_username(username2)}")  # Output: False
    
    # 4. ASCII control characters example
    control_chars = ''.join(chr(i) for i in range(32)) + chr(127)
    print(f"Control characters are ASCII: {control_chars.isascii()}")  # Output: True


# ====================================================================================================
# str.isdecimal()
# ====================================================================================================
def demo_isdecimal():
    """
    str.isdecimal(): Returns True if all characters in the string are decimal characters.
    
    Syntax:
        string.isdecimal()
    
    Returns:
        bool: True if all characters are decimal digits (0-9) and there is at least one character, False otherwise
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'123'.isdecimal(): {'123'.isdecimal()}")  # Output: '123'.isdecimal(): True
    print(f"'123.45'.isdecimal(): {'123.45'.isdecimal()}")  # Output: '123.45'.isdecimal(): False (decimal point is not a decimal digit)
    
    # Empty string
    print(f"''.isdecimal(): {''.isdecimal()}")  # Output: ''.isdecimal(): False (empty string)
    
    # Letters and other characters
    print(f"'123abc'.isdecimal(): {'123abc'.isdecimal()}")  # Output: '123abc'.isdecimal(): False (contains letters)
    print(f"'１２３'.isdecimal(): {'１２３'.isdecimal()}")  # Output: '１２３'.isdecimal(): True (these are Unicode fullwidth digits)
    
    # Various numeric representations
    print("\nVarious numeric representations:")
    # Regular digits (0-9)
    print(f"'0123456789'.isdecimal(): {'0123456789'.isdecimal()}")  # True
    
    # Superscript/subscript digits
    print(f"'²³'.isdecimal(): {'²³'.isdecimal()}")  # False - superscript digits aren't decimal
    
    # Fraction characters
    print(f"'½'.isdecimal(): {'½'.isdecimal()}")  # False - fractions aren't decimal digits
    
    # Numeric characters from other languages
    print(f"'٣٤٥'.isdecimal(): {'٣٤٥'.isdecimal()}")  # True - Arabic-Indic digits
    print(f"'二三四'.isdecimal(): {'二三四'.isdecimal()}")  # False - Chinese/Japanese numerals aren't decimal digits
    
    # Comparison with isdigit() and isnumeric()
    print("\nComparison with isdigit() and isnumeric():")
    examples = ['123', '١٢٣', '½', '²', '二', '⅔', '']
    
    print("String    | isdecimal() | isdigit() | isnumeric()")
    print("----------|-------------|-----------|------------")
    for ex in examples:
        print(f"'{ex}'".ljust(10), "|", 
              str(ex.isdecimal()).ljust(13), "|",
              str(ex.isdigit()).ljust(11), "|",
              str(ex.isnumeric()))
    
    # Practical examples
    # 1. Validating user input for a numeric field
    user_input = "12345"
    if user_input.isdecimal():
        number = int(user_input)
        print(f"Valid number: {number}")
    else:
        print(f"'{user_input}' is not a valid decimal number")
    
    # 2. Filtering decimal numbers from a list
    items = ["123", "abc", "456", "12.34", "-789", "١٢٣"]
    decimal_numbers = [item for item in items if item.isdecimal()]
    print(f"Decimal numbers: {decimal_numbers}")  # Output: Decimal numbers: ['123', '456', '١٢٣']
    
    # 3. Safe conversion to integer with different numeric systems
    def safe_int_convert(s):
        if s.isdecimal():
            return int(s)
        return None
    
    values = ["123", "١٢٣", "IV", "四"]
    converted = [safe_int_convert(v) for v in values]
    print(f"Converted values: {converted}")  # Output: Converted values: [123, 123, None, None]


# ====================================================================================================
# str.isdigit()
# ====================================================================================================
def demo_isdigit():
    """
    str.isdigit(): Returns True if all characters in the string are digits.
    
    Syntax:
        string.isdigit()
    
    Returns:
        bool: True if all characters are digits and there is at least one character, False otherwise
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'123'.isdigit(): {'123'.isdigit()}")  # Output: '123'.isdigit(): True
    print(f"'123.45'.isdigit(): {'123.45'.isdigit()}")  # Output: '123.45'.isdigit(): False (decimal point is not a digit)
    
    # Empty string
    print(f"''.isdigit(): {''.isdigit()}")  # Output: ''.isdigit(): False (empty string)
    
    # Letters and other characters
    print(f"'123abc'.isdigit(): {'123abc'.isdigit()}")  # Output: '123abc'.isdigit(): False (contains letters)
    
    # Various numeric representations
    print("\nVarious numeric representations:")
    # Regular digits (0-9)
    print(f"'0123456789'.isdigit(): {'0123456789'.isdigit()}")  # True
    
    # Superscript/subscript digits
    print(f"'²³'.isdigit(): {'²³'.isdigit()}")  # True - superscript digits are considered digits
    
    # Circled digits
    print(f"'①②③'.isdigit(): {'①②③'.isdigit()}")  # False - these are not considered digits by isdigit()
    
    # Fraction characters
    print(f"'½'.isdigit(): {'½'.isdigit()}")  # False - fractions aren't digits
    
    # Fullwidth digits
    print(f"'１２３'.isdigit(): {'１２３'.isdigit()}")  # True - fullwidth digits are considered digits
    
    # Arabic-Indic digits
    print(f"'٣٤٥'.isdigit(): {'٣٤٥'.isdigit()}")  # True - Arabic-Indic digits are considered digits
    
    # Negative numbers
    print(f"'-123'.isdigit(): {'-123'.isdigit()}")  # False - the minus sign is not a digit
    
    # Comparison with isdecimal() and isnumeric()
    print("\nComparison with isdecimal() and isnumeric():")
    examples = ['123', '١٢٣', '½', '²', '②', '']
    
    print("String    | isdecimal() | isdigit() | isnumeric()")
    print("----------|-------------|-----------|------------")
    for ex in examples:
        print(f"'{ex}'".ljust(10), "|", 
              str(ex.isdecimal()).ljust(13), "|",
              str(ex.isdigit()).ljust(11), "|",
              str(ex.isnumeric()))
    
    # Practical examples
    # 1. Checking if a string can be safely converted to an integer
    def is_safe_int(s):
        return s.isdigit() and len(s) > 0
    
    inputs = ["123", "abc", "", "12.34", "-456", "⁵"]
    for inp in inputs:
        if is_safe_int(inp):
            print(f"'{inp}' can be safely converted to int")
        else:
            print(f"'{inp}' cannot be safely converted to int")
    
    # 2. Filtering numeric strings from a list
    items = ["123", "abc", "456", "12.34", "-789", "²³"]
    digit_only = [item for item in items if item.isdigit()]
    print(f"Digit-only strings: {digit_only}")  # Output: Digit-only strings: ['123', '456', '²³']
    
    # 3. Validating PIN code
    def is_valid_pin(pin):
        return pin.isdigit() and (len(pin) == 4 or len(pin) == 6)
    
    pins = ["1234", "123456", "123", "12345", "abcd"]
    for pin in pins:
        print(f"PIN '{pin}' is valid: {is_valid_pin(pin)}")


# ====================================================================================================
# str.isidentifier()
# ====================================================================================================
def demo_isidentifier():
    """
    str.isidentifier(): Returns True if the string is a valid identifier in Python.
    
    Syntax:
        string.isidentifier()
    
    Returns:
        bool: True if the string is a valid Python identifier, False otherwise
    
    Notes:
        - A valid identifier starts with a letter or underscore (_), followed by any number of
          letters, numbers, or underscores.
        - Keywords (like 'if', 'else', 'def') may be valid identifiers but cannot be used as variable
          names in Python.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'variable'.isidentifier(): {'variable'.isidentifier()}")  # Output: 'variable'.isidentifier(): True
    print(f"'_private'.isidentifier(): {'_private'.isidentifier()}")  # Output: '_private'.isidentifier(): True
    print(f"'variable_1'.isidentifier(): {'variable_1'.isidentifier()}")  # Output: 'variable_1'.isidentifier(): True
    
    # Invalid identifiers
    print(f"'123var'.isidentifier(): {'123var'.isidentifier()}")  # Output: '123var'.isidentifier(): False (starts with a digit)
    print(f"'var-name'.isidentifier(): {'var-name'.isidentifier()}")  # Output: 'var-name'.isidentifier(): False (contains hyphen)
    print(f"'var name'.isidentifier(): {'var name'.isidentifier()}")  # Output: 'var name'.isidentifier(): False (contains space)
    print(f"'var!'.isidentifier(): {'var!'.isidentifier()}")  # Output: 'var!'.isidentifier(): False (contains special character)
    
    # Empty string
    print(f"''.isidentifier(): {''.isidentifier()}")  # Output: ''.isidentifier(): False (empty string)
    
    # Keywords
    print(f"'if'.isidentifier(): {'if'.isidentifier()}")  # Output: 'if'.isidentifier(): True (valid identifier but Python keyword)
    print(f"'class'.isidentifier(): {'class'.isidentifier()}")  # Output: 'class'.isidentifier(): True (valid identifier but Python keyword)
    
    # Unicode support
    print(f"'π'.isidentifier(): {'π'.isidentifier()}")  # Output: 'π'.isidentifier(): True (Unicode letter)
    print(f"'변수'.isidentifier(): {'변수'.isidentifier()}")  # Output: '변수'.isidentifier(): True (Korean characters)
    
    # Checking if a keyword
    import keyword
    
    def is_valid_variable_name(name):
        return name.isidentifier() and not keyword.iskeyword(name)
    
    names = ["variable", "if", "_private", "123var", "class", "for", "var_1", "π"]
    for name in names:
        if is_valid_variable_name(name):
            print(f"'{name}' is a valid variable name")
        else:
            if name.isidentifier():
                print(f"'{name}' is a valid identifier but a Python keyword")
            else:
                print(f"'{name}' is not a valid identifier")
    
    # Practical examples
    # 1. Checking dynamic attribute names
    class DynamicObject:
        def __init__(self):
            pass
        
        def set_attribute(self, name, value):
            if name.isidentifier():
                setattr(self, name, value)
                return True
            return False
    
    obj = DynamicObject()
    attrs = ["name", "age", "user-id", "123value"]
    
    for attr in attrs:
        if obj.set_attribute(attr, "test"):
            print(f"Set attribute '{attr}' successfully")
        else:
            print(f"Could not set attribute '{attr}' (invalid identifier)")
    
    # 2. Checking if user input could be a valid variable name
    user_input = "my_var_1"
    if user_input.isidentifier() and not keyword.iskeyword(user_input):
        print(f"'{user_input}' can be used as a variable name")
    else:
        print(f"'{user_input}' cannot be used as a variable name")


# ====================================================================================================
# str.islower()
# ====================================================================================================
def demo_islower():
    """
    str.islower(): Returns True if all cased characters in the string are lowercase.
    
    Syntax:
        string.islower()
    
    Returns:
        bool: True if all cased characters are lowercase and there is at least one cased character,
              False otherwise
    
    Notes:
        - Numbers, symbols, and spaces don't affect the result since they are not cased characters.
        - The string must contain at least one cased character to return True.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'hello'.islower(): {'hello'.islower()}")  # Output: 'hello'.islower(): True
    print(f"'Hello'.islower(): {'Hello'.islower()}")  # Output: 'Hello'.islower(): False (contains uppercase 'H')
    print(f"'hello world'.islower(): {'hello world'.islower()}")  # Output: 'hello world'.islower(): True (spaces don't affect the result)
    
    # With numbers and symbols
    print(f"'hello123'.islower(): {'hello123'.islower()}")  # Output: 'hello123'.islower(): True (numbers don't affect the result)
    print(f"'hello!@#'.islower(): {'hello!@#'.islower()}")  # Output: 'hello!@#'.islower(): True (symbols don't affect the result)
    
    # Edge cases
    print(f"''.islower(): {''.islower()}")  # Output: ''.islower(): False (empty string)
    print(f"'123'.islower(): {'123'.islower()}")  # Output: '123'.islower(): False (no cased characters)
    print(f"'!@#'.islower(): {'!@#'.islower()}")  # Output: '!@#'.islower(): False (no cased characters)
    
    # Unicode support
    print(f"'café'.islower(): {'café'.islower()}")  # Output: 'café'.islower(): True
    print(f"'Café'.islower(): {'Café'.islower()}")  # Output: 'Café'.islower(): False
    
    # Practical examples
    # 1. Checking case consistency in user input
    username = "johndoe"
    if username.islower():
        print(f"Username '{username}' is all lowercase (consistent format)")
    else:
        print(f"Username '{username}' is not all lowercase")
    
    # 2. Converting text to lowercase if needed
    def ensure_lowercase(text):
        if not text.islower():
            return text.lower()
        return text  # Already lowercase
    
    mixed_case = "Mixed CASE Text"
    consistent = ensure_lowercase(mixed_case)
    print(f"Original: '{mixed_case}' → Consistent: '{consistent}'")
    
    # 3. Checking if a password has a mix of cases (simple password strength check)
    def has_mixed_case(password):
        return not password.islower() and not password.isupper()
    
    passwords = ["password", "PASSWORD", "Password", "pass123"]
    for password in passwords:
        if has_mixed_case(password):
            print(f"Password '{password}' has mixed case (stronger)")
        else:
            print(f"Password '{password}' does not have mixed case (weaker)")
    
    # 4. Parsing case-sensitive commands
    command = "help"
    if command.islower():
        print(f"Executing standard command: {command}")
    else:
        print(f"Executing special command variation: {command}")


# ====================================================================================================
# str.isnumeric()
# ====================================================================================================
def demo_isnumeric():
    """
    str.isnumeric(): Returns True if all characters in the string are numeric characters.
    
    Syntax:
        string.isnumeric()
    
    Returns:
        bool: True if all characters are numeric and there is at least one character, False otherwise
    
    Notes:
        - More inclusive than isdigit() and isdecimal(), includes characters like ½, ③, etc.
        - Includes digits, numeric literals, and all characters with the Unicode numeric value property.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'123'.isnumeric(): {'123'.isnumeric()}")  # Output: '123'.isnumeric(): True
    print(f"'123.45'.isnumeric(): {'123.45'.isnumeric()}")  # Output: '123.45'.isnumeric(): False (decimal point is not numeric)
    
    # Empty string
    print(f"''.isnumeric(): {''.isnumeric()}")  # Output: ''.isnumeric(): False (empty string)
    
    # Unicode numeric characters
    print("\nUnicode numeric examples:")
    # Traditional digits (0-9)
    print(f"'0123456789'.isnumeric(): {'0123456789'.isnumeric()}")  # True
    
    # Superscript/subscript numbers
    print(f"'²³'.isnumeric(): {'²³'.isnumeric()}")  # True
    
    # Fractions
    print(f"'½⅔¾'.isnumeric(): {'½⅔¾'.isnumeric()}")  # True
    
    # Circled numbers
    print(f"'①②③'.isnumeric(): {'①②③'.isnumeric()}")  # True
    
    # Roman numerals
    print(f"'ⅠⅡⅢ'.isnumeric(): {'ⅠⅡⅢ'.isnumeric()}")  # True
    
    # Arabic numerals
    print(f"'٠١٢٣٤'.isnumeric(): {'٠١٢٣٤'.isnumeric()}")  # True
    
    # Chinese/Japanese numerals
    print(f"'一二三四五'.isnumeric(): {'一二三四五'.isnumeric()}")  # True
    
    # Bengali numerals
    print(f"'০১২৩'.isnumeric(): {'০১২৩'.isnumeric()}")  # True
    
    # Mixed characters (not numeric)
    print(f"'123abc'.isnumeric(): {'123abc'.isnumeric()}")  # False
    print(f"'1.23'.isnumeric(): {'1.23'.isnumeric()}")  # False
    print(f"'-123'.isnumeric(): {'-123'.isnumeric()}")  # False
    
    # Comparison with isdigit() and isdecimal()
    print("\nComparison with isdecimal() and isdigit():")
    examples = ['123', '١٢٣', '½', '²', '二三四', 'ⅠⅡⅢ', '①②③', '']
    
    print("String    | isdecimal() | isdigit() | isnumeric()")
    print("----------|-------------|-----------|------------")
    for ex in examples:
        print(f"'{ex}'".ljust(10), "|", 
              str(ex.isdecimal()).ljust(13), "|",
              str(ex.isdigit()).ljust(11), "|",
              str(ex.isnumeric()))
    
    # Practical examples
    # 1. Supporting international numeric input
    def is_numeric_input(s):
        return s.isnumeric()
    
    inputs = ["123", "١٢٣", "一二三", "½", "IV"]
    for inp in inputs:
        if is_numeric_input(inp):
            print(f"'{inp}' is considered numeric input")
        else:
            print(f"'{inp}' is not considered numeric input")
    
    # 2. Converting various numeric representations to integers (when possible)
    def safe_int_convert(s):
        # Warning: This works for some but not all numeric characters
        # For example, fractions and roman numerals need special handling
        numeric_map = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '零': 0,
            '١': 1, '٢': 2, '٣': 3, '٤': 4, '٥': 5,
            '٦': 6, '٧': 7, '٨': 8, '٩': 9, '٠': 0,
        }
        
        if not s.isnumeric():
            return None
            
        try:
            return int(s)
        except ValueError:
            # Handle non-convertible numerics
            if all(c in numeric_map for c in s):
                result = 0
                for c in s:
                    result = result * 10 + numeric_map[c]
                return result
            return None
    
    test_nums = ["123", "١٢٣", "一二三", "④⑤⑥"]
    for num in test_nums:
        conv = safe_int_convert(num)
        if conv is not None:
            print(f"'{num}' converted to: {conv}")
        else:
            print(f"'{num}' could not be converted")


# ====================================================================================================
# str.isprintable()
# ====================================================================================================
def demo_isprintable():
    """
    str.isprintable(): Returns True if all characters in the string are printable or if the string is empty.
    
    Syntax:
        string.isprintable()
    
    Returns:
        bool: True if all characters are printable or if the string is empty, False otherwise
    
    Notes:
        - Printable characters are those characters that are not a control character.
        - Control characters are characters that do not represent a written symbol (e.g., \n, \t, etc.).
        - Space is considered printable, but tab (\t), newline (\n), carriage return (\r), etc. are not.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'Hello, World!'.isprintable(): {'Hello, World!'.isprintable()}")  # Output: 'Hello, World!'.isprintable(): True
    print(f"'Hello\\nWorld'.isprintable(): {'Hello\nWorld'.isprintable()}")  # Output: 'Hello\nWorld'.isprintable(): False (contains newline)
    
    # Control characters
    print(f"'\\t\\r\\n'.isprintable(): {'\t\r\n'.isprintable()}")  # Output: '\t\r\n'.isprintable(): False (control characters)
    
    # Space and printable symbols
    print(f"' !@#$%^&*()'.isprintable(): {' !@#$%^&*()'.isprintable()}")  # Output: ' !@#$%^&*()'.isprintable(): True (space and symbols are printable)
    
    # Empty string
    print(f"''.isprintable(): {''.isprintable()}")  # Output: ''.isprintable(): True (empty string is considered printable)
    
    # Unicode characters
    print(f"'Café 123'.isprintable(): {'Café 123'.isprintable()}")  # Output: 'Café 123'.isprintable(): True (accented letters are printable)
    print(f"'こんにちは'.isprintable(): {'こんにちは'.isprintable()}")  # Output: 'こんにちは'.isprintable(): True (Japanese characters are printable)
    print(f"'😀👍'.isprintable(): {'😀👍'.isprintable()}")  # Output: '😀👍'.isprintable(): True (emojis are printable)
    
    # Unprintable control characters
    print("\nUnprintable characters examples:")
    
    # Create a string with all ASCII control characters
    control_chars = ''.join(chr(i) for i in range(32)) + chr(127)
    print(f"ASCII control characters (0-31, 127) are printable: {control_chars.isprintable()}")  # False
    
    # Escape sequences examples
    escapes = {
        '\\n': '\n',
        '\\t': '\t',
        '\\r': '\r',
        '\\b': '\b',
        '\\f': '\f',
        '\\v': '\v'
    }
    
    for name, char in escapes.items():
        print(f"'{name}' is printable: {char.isprintable()}")  # All should be False
    
    # Practical examples
    # 1. Checking if a string can be printed without special handling
    def is_safely_printable(text):
        return text.isprintable()
    
    texts = ["Hello", "Line 1\nLine 2", "Tab\tIndented", "Bell\a"]
    for text in texts:
        if is_safely_printable(text):
            print(f"'{text}' can be safely printed as-is")
        else:
            print(f"'{text}' contains unprintable characters (needs escaping)")
    
    # 2. Filtering out unprintable characters
    def remove_unprintable(text):
        return ''.join(char for char in text if char.isprintable())
    
    mixed_text = "Hello\nWorld\tWith\rUnprintable\bChars"
    clean_text = remove_unprintable(mixed_text)
    print(f"Original: '{mixed_text}' → Cleaned: '{clean_text}'")
    
    # 3. Validating user input for display
    def validate_display_input(text):
        if not text.isprintable():
            return False, "Input contains characters that cannot be displayed"
        return True, "Input is valid for display"
    
    inputs = ["Hello, World!", "Notification\a"]
    for inp in inputs:
        valid, message = validate_display_input(inp)
        print(f"'{inp}' - {message}")


# ====================================================================================================
# str.isspace()
# ====================================================================================================
def demo_isspace():
    """
    str.isspace(): Returns True if all characters in the string are whitespace.
    
    Syntax:
        string.isspace()
    
    Returns:
        bool: True if all characters are whitespace and there is at least one character, False otherwise
    
    Notes:
        - Whitespace characters include: space, tab, newline, carriage return, form feed, and vertical tab.
        - The string must contain at least one character to return True.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"' '.isspace(): {' '.isspace()}")  # Output: ' '.isspace(): True (space)
    print(f"'\\t'.isspace(): {'\t'.isspace()}")  # Output: '\t'.isspace(): True (tab)
    print(f"'\\n'.isspace(): {'\n'.isspace()}")  # Output: '\n'.isspace(): True (newline)
    print(f"'\\r'.isspace(): {'\r'.isspace()}")  # Output: '\r'.isspace(): True (carriage return)
    
    # Multiple whitespace characters
    print(f"' \\t\\n\\r'.isspace(): {' \t\n\r'.isspace()}")  # Output: ' \t\n\r'.isspace(): True (multiple whitespace)
    
    # Non-whitespace characters
    print(f"'Hello'.isspace(): {'Hello'.isspace()}")  # Output: 'Hello'.isspace(): False (contains letters)
    print(f"' A '.isspace(): {' A '.isspace()}")  # Output: ' A '.isspace(): False (contains a letter)
    print(f"'\\t123'.isspace(): {'\t123'.isspace()}")  # Output: '\t123'.isspace(): False (contains numbers)
    
    # Empty string
    print(f"''.isspace(): {''.isspace()}")  # Output: ''.isspace(): False (empty string)
    
    # Unicode whitespace
    print(f"'\\u2003'.isspace(): {'\u2003'.isspace()}")  # Output: '\u2003'.isspace(): True (Unicode EM SPACE)
    print(f"'\\u2000'.isspace(): {'\u2000'.isspace()}")  # Output: '\u2000'.isspace(): True (Unicode EN QUAD)
    
    # All whitespace characters
    whitespace_chars = [
        ' ',    # Space
        '\t',   # Tab
        '\n',   # Newline
        '\r',   # Carriage Return
        '\f',   # Form Feed
        '\v',   # Vertical Tab
        '\u00A0',  # Non-breaking space
        '\u2000',  # En Quad
        '\u2001',  # Em Quad
        '\u2002',  # En Space
        '\u2003',  # Em Space
        '\u2004',  # Three-Per-Em Space
        '\u2005',  # Four-Per-Em Space
        '\u2006',  # Six-Per-Em Space
        '\u2007',  # Figure Space
        '\u2008',  # Punctuation Space
        '\u2009',  # Thin Space
        '\u200A',  # Hair Space
        '\u202F',  # Narrow No-Break Space
        '\u205F',  # Medium Mathematical Space
        '\u3000',  # Ideographic Space
    ]
    
    print("\nTesting all whitespace characters:")
    for i, ws in enumerate(whitespace_chars):
        print(f"Whitespace #{i+1}: '{ws}' is space: {ws.isspace()}")
    
    # Practical examples
    # 1. Checking if a string contains only whitespace
    def is_empty_or_whitespace(text):
        return not text or text.isspace()
    
    texts = ["", " ", "\t\n", "  hello  ", "   "]
    for text in texts:
        if is_empty_or_whitespace(text):
            print(f"'{text}' is empty or whitespace only")
        else:
            print(f"'{text}' contains non-whitespace characters")
    
    # 2. Removing empty lines from multi-line text
    def remove_empty_lines(text):
        return '\n'.join(line for line in text.split('\n') if not is_empty_or_whitespace(line))
    
    multiline = """
    Line 1
    
    Line 2
      
    Line 3
    """
    
    cleaned = remove_empty_lines(multiline)
    print(f"\nOriginal multiline text:\n'{multiline}'")
    print(f"\nCleaned multiline text:\n'{cleaned}'")
    
    # 3. Validating user input
    def validate_input(user_input):
        if is_empty_or_whitespace(user_input):
            return False, "Input cannot be empty or whitespace only"
        return True, "Input is valid"
    
    inputs = ["Hello", "", "   ", "\t\n"]
    for inp in inputs:
        valid, message = validate_input(inp)
        print(f"Input '{inp}' - {message}")


# ====================================================================================================
# str.istitle()
# ====================================================================================================
def demo_istitle():
    """
    str.istitle(): Returns True if the string is titlecased.
    
    Syntax:
        string.istitle()
    
    Returns:
        bool: True if the string is titlecased (each word starts with an uppercase letter followed by lowercase),
              False otherwise
    
    Notes:
        - A word is defined as a sequence of alphabetic characters.
        - Each word must start with an uppercase letter followed by lowercase letters.
        - The string must have at least one character to return True.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'Hello World'.istitle(): {'Hello World'.istitle()}")  # Output: 'Hello World'.istitle(): True
    print(f"'Hello world'.istitle(): {'Hello world'.istitle()}")  # Output: 'Hello world'.istitle(): False (lowercase 'w' in 'world')
    print(f"'HELLO WORLD'.istitle(): {'HELLO WORLD'.istitle()}")  # Output: 'HELLO WORLD'.istitle(): False (all uppercase)
    
    # With punctuation and numbers
    print(f"'Hello, World! 123'.istitle(): {'Hello, World! 123'.istitle()}")  # Output: 'Hello, World! 123'.istitle(): True
    print(f"'Hello, world! 123'.istitle(): {'Hello, world! 123'.istitle()}")  # Output: 'Hello, world! 123'.istitle(): False
    
    # Edge cases
    print(f"''.istitle(): {''.istitle()}")  # Output: ''.istitle(): False (empty string)
    print(f"'A'.istitle(): {'A'.istitle()}")  # Output: 'A'.istitle(): True (single uppercase letter)
    print(f"'a'.istitle(): {'a'.istitle()}")  # Output: 'a'.istitle(): False (single lowercase letter)
    print(f"'123'.istitle(): {'123'.istitle()}")  # Output: '123'.istitle(): False (only numbers)
    
    # Mixed case words
    print(f"'MixedCase Word'.istitle(): {'MixedCase Word'.istitle()}")  # Output: 'MixedCase Word'.istitle(): False
    
    # Hyphenated words
    print(f"'First-Name Last-Name'.istitle(): {'First-Name Last-Name'.istitle()}")  # Output: 'First-Name Last-Name'.istitle(): True
    
   
    # Unicode support
    print(f"'Café Olé'.istitle(): {'Café Olé'.istitle()}")  # Output: 'Café Olé'.istitle(): True
    
    # Practical examples
    # 1. Proper formatting of titles
    titles = [
        "the lord of the rings",
        "THE GREAT GATSBY",
        "To Kill A Mockingbird",
        "harry potter and the sorcerer's stone"
    ]
    
    for title in titles:
        title_case = title.title()
        print(f"Original: '{title}' → Title Case: '{title_case}'")
        print(f"Is title case: {title_case.istitle()}")
    
    # 2. Validating names
    def is_properly_formatted_name(name):
        return name.istitle() and all(c.isalpha() or c.isspace() or c == '-' or c == "'" for c in name)
    
    names = ["John Smith", "john smith", "John-Paul", "John smith", "John123"]
    for name in names:
        if is_properly_formatted_name(name):
            print(f"'{name}' is properly formatted")
        else:
            print(f"'{name}' is not properly formatted")
    
    # 3. Correcting title case
    def ensure_title_case(text):
        if not text.istitle():
            return text.title()
        return text  # Already in title case
    
    examples = ["mixed CASE text", "Already Title Case"]
    for example in examples:
        corrected = ensure_title_case(example)
        print(f"Original: '{example}' → Corrected: '{corrected}'")


# ====================================================================================================
# str.isupper()
# ====================================================================================================
def demo_isupper():
    """
    str.isupper(): Returns True if all cased characters in the string are uppercase.
    
    Syntax:
        string.isupper()
    
    Returns:
        bool: True if all cased characters are uppercase and there is at least one cased character,
              False otherwise
    
    Notes:
        - Numbers, symbols, and spaces don't affect the result since they are not cased characters.
        - The string must contain at least one cased character to return True.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    """
    # Basic examples
    print(f"'HELLO'.isupper(): {'HELLO'.isupper()}")  # Output: 'HELLO'.isupper(): True
    print(f"'Hello'.isupper(): {'Hello'.isupper()}")  # Output: 'Hello'.isupper(): False (contains lowercase 'ello')
    print(f"'HELLO WORLD'.isupper(): {'HELLO WORLD'.isupper()}")  # Output: 'HELLO WORLD'.isupper(): True (spaces don't affect the result)
    
    # With numbers and symbols
    print(f"'HELLO123'.isupper(): {'HELLO123'.isupper()}")  # Output: 'HELLO123'.isupper(): True (numbers don't affect the result)
    print(f"'HELLO!@#'.isupper(): {'HELLO!@#'.isupper()}")  # Output: 'HELLO!@#'.isupper(): True (symbols don't affect the result)
    
    # Edge cases
    print(f"''.isupper(): {''.isupper()}")  # Output: ''.isupper(): False (empty string)
    print(f"'123'.isupper(): {'123'.isupper()}")  # Output: '123'.isupper(): False (no cased characters)
    print(f"'!@#'.isupper(): {'!@#'.isupper()}")  # Output: '!@#'.isupper(): False (no cased characters)
    
    # Unicode support
    print(f"'CAFÉ'.isupper(): {'CAFÉ'.isupper()}")  # Output: 'CAFÉ'.isupper(): True
    print(f"'CAFé'.isupper(): {'CAFé'.isupper()}")  # Output: 'CAFé'.isupper(): False
    
    # Practical examples
    # 1. Checking if text is all caps (e.g., to detect "shouting" in messages)
    message = "PLEASE HELP ME NOW!"
    if message.isupper():
        print(f"Message '{message}' appears to be shouting (all caps)")
    else:
        print(f"Message '{message}' uses normal case")
    
    # 2. Ensuring consistent formatting for codes or identifiers
    def ensure_uppercase(code):
        if not code.isupper():
            return code.upper()
        return code  # Already uppercase
    
    codes = ["ABC123", "def456", "GHI-789"]
    for code in codes:
        standardized = ensure_uppercase(code)
        print(f"Original: '{code}' → Standardized: '{standardized}'")
    
    # 3. Detecting emphasized text
    def is_emphasized(text):
        return text.isupper() or text.startswith('*') and text.endswith('*')
    
    texts = ["normal text", "IMPORTANT", "*special*", "Partly UPPERCASE"]
    for text in texts:
        if is_emphasized(text):
            print(f"'{text}' appears to be emphasized")
        else:
            print(f"'{text}' is normal text")
    
    # 4. Implementing case insensitive comparisons
    def case_insensitive_equal(str1, str2):
        return str1.upper() == str2.upper()
    
    pairs = [("hello", "HELLO"), ("WORLD", "world"), ("Python", "Java")]
    for str1, str2 in pairs:
        if case_insensitive_equal(str1, str2):
            print(f"'{str1}' and '{str2}' are equal (case-insensitive)")
        else:
            print(f"'{str1}' and '{str2}' are different")


# ====================================================================================================
# str.join()
# ====================================================================================================
def demo_join():
    """
    str.join(iterable): Returns a string that is the concatenation of the strings in the iterable, 
                         with the given string as a separator.
    
    Syntax:
        separator.join(iterable)
    
    Parameters:
        iterable: An iterable object containing strings to be joined
    
    Returns:
        str: A new string with elements from the iterable separated by the separator
    
    Raises:
        TypeError: If the iterable contains non-string elements
    
    Time Complexity: O(n) where n is the total length of all strings in the iterable
    Space Complexity: O(n) for the new string
    """
    # Basic examples
    separator = ", "
    words = ["apple", "banana", "cherry"]
    joined = separator.join(words)
    print(f"Join with '{separator}': '{joined}'")  # Output: Join with ', ': 'apple, banana, cherry'
    
    # Empty separator
    empty_sep = "".join(words)
    print(f"Join with empty separator: '{empty_sep}'")  # Output: Join with empty separator: 'applebananacherry'
    
    # Multicharacter separator
    multi_sep = " | ".join(words)
    print(f"Join with ' | ': '{multi_sep}'")  # Output: Join with ' | ': 'apple | banana | cherry'
    
    # Newline separator
    newline_sep = "\n".join(words)
    print(f"Join with newline:\n'{newline_sep}'")
    
    # Joining with different separators
    print("\nJoining with different separators:")
    separators = ["", ",", " - ", "::"]
    for sep in separators:
        print(f"'{sep}'.join({words}): '{sep.join(words)}'")
    
    # Joining an empty iterable
    empty_list = []
    print(f"Join empty list: '{','.join(empty_list)}'")  # Output: Join empty list: ''
    
    # Join string characters
    word = "Python"
    char_joined = "-".join(word)
    print(f"Join characters of '{word}': '{char_joined}'")  # Output: Join characters of 'Python': 'P-y-t-h-o-n'
    
    # Join from a set (unordered)
    fruits_set = {"apple", "banana", "cherry"}
    set_joined = ", ".join(fruits_set)
    print(f"Join from set: '{set_joined}'")  # Output will vary due to set's unordered nature
    
    # Join from a dictionary (joins the keys)
    fruits_dict = {"apple": 1, "banana": 2, "cherry": 3}
    dict_joined = ", ".join(fruits_dict)
    print(f"Join from dict keys: '{dict_joined}'")  # Output will vary due to dict's unordered nature
    
    # Join values from a dictionary
    values_joined = ", ".join(str(value) for value in fruits_dict.values())
    print(f"Join from dict values: '{values_joined}'")  # Output may vary
    
    # Error case: join non-string elements
    try:
        # This will raise TypeError
        numbers = [1, 2, 3]
        ",".join(numbers)
    except TypeError as e:
        print(f"Error joining non-strings: {e}")
    
    # Converting non-string elements to strings before joining
    numbers = [1, 2, 3]
    numbers_str = [str(num) for num in numbers]
    numbers_joined = ", ".join(numbers_str)
    print(f"Join numbers after conversion: '{numbers_joined}'")  # Output: Join numbers after conversion: '1, 2, 3'
    
    # Alternative with generator expression
    numbers = [1, 2, 3]
    numbers_joined = ", ".join(str(num) for num in numbers)
    print(f"Join with generator expression: '{numbers_joined}'")  # Output: Join with generator expression: '1, 2, 3'
    
    # Practical examples
    # 1. Building a CSV line
    data = ["John", "Doe", "30", "New York"]
    csv_line = ",".join(data)
    print(f"CSV line: '{csv_line}'")  # Output: CSV line: 'John,Doe,30,New York'
    
    # 2. Building a URL query string
    params = {
        "name": "John Smith",
        "age": "30",
        "city": "New York"
    }
    query_parts = [f"{key}={value}" for key, value in params.items()]
    query_string = "&".join(query_parts)
    print(f"Query string: '{query_string}'")  # Output: Query string: 'name=John Smith&age=30&city=New York'
    
    # 3. Creating a formatted list
    items = ["apples", "bananas", "cherries"]
    formatted_list = "• " + "\n• ".join(items)
    print(f"Formatted list:\n{formatted_list}")


# ====================================================================================================
# str.ljust()
# ====================================================================================================
def demo_ljust():
    """
    str.ljust(width[, fillchar]): Returns a left-justified string of specified width.
    
    Syntax:
        string.ljust(width[, fillchar])
    
    Parameters:
        width (int): The total width of the resulting string
        fillchar (str, optional): The character to pad with. Defaults to space.
    
    Returns:
        str: A new string that is left-justified within a field of the specified width
    
    Raises:
        TypeError: If width is not an integer, or fillchar is not a character
    
    Time Complexity: O(n) where n is the width
    Space Complexity: O(n) for the new string
    """
    # Basic example
    text = "Hello"
    left_justified = text.ljust(10)
    print(f"Original: '{text}' → Left justified (width=10): '{left_justified}'")  # Output: Original: 'Hello' → Left justified (width=10): 'Hello     '
    
    # Using custom fill character
    left_justified_star = text.ljust(10, '*')
    print(f"Left justified with '*': '{left_justified_star}'")  # Output: Left justified with '*': 'Hello*****'
    
    # When width is less than or equal to the length of the string
    print(f"Width equal to length: '{text.ljust(5)}'")  # Output: Width equal to length: 'Hello'
    print(f"Width less than length: '{text.ljust(3)}'")  # Output: Width less than length: 'Hello'
    
    # Various fill characters
    print("\nVarious fill characters:")
    fill_chars = [' ', '-', '.', '=']
    for char in fill_chars:
        print(f"'{text.ljust(10, char)}'")
    
    # Empty string
    empty = ""
    print(f"Empty string: '{empty.ljust(5, '_')}'")  # Output: Empty string: '_____'
    
    # Unicode/emoji support
    emoji = "🐍"
    print(f"Unicode: '{emoji.ljust(5, '.')}'")  # Output: Unicode: '🐍....'
    
    # Exception cases
    try:
        # TypeError: ljust() argument 1 must be int, not str
        text.ljust("10")
    except TypeError as e:
        print(f"Error with non-integer width: {e}")
    
    try:
        # TypeError: ljust() argument 2 must be str of length 1, not str
        text.ljust(10, "**")
    except TypeError as e:
        print(f"Error with multi-character fillchar: {e}")
    
    # Practical examples
    # 1. Creating a simple text table
    def print_table(headers, data):
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in data:
            for i, value in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(value)))
        
        # Print headers
        header_line = "| "
        for i, header in enumerate(headers):
            header_line += header.ljust(col_widths[i]) + " | "
        print(header_line)
        
        # Print separator
        separator = "+-" + "-+-".join("-" * width for width in col_widths) + "-+"
        print(separator)
        
        # Print data
        for row in data:
            row_line = "| "
            for i, value in enumerate(row):
                row_line += str(value).ljust(col_widths[i]) + " | "
            print(row_line)
    
    # Example usage
    headers = ["Name", "Age", "City"]
    data = [
        ["Alice", 30, "New York"],
        ["Bob", 25, "San Francisco"],
        ["Charlie", 35, "Los Angeles"]
    ]
    
    print("\nSimple text table:")
    print_table(headers, data)
    
    # 2. Creating a formatted file listing
    files = [
        {"name": "document.pdf", "size": 1024},
        {"name": "image.png", "size": 2048},
        {"name": "spreadsheet.xlsx", "size": 512}
    ]
    
    print("\nFormatted file listing:")
    for file in files:
        line = file["name"].ljust(20) + str(file["size"]).rjust(10) + " bytes"
        print(line)
    
    # 3. Padding numbers to specific width (e.g., for formatting codes)
    codes = ["A1", "B12", "C123"]
    padded_codes = [code.ljust(5, '0') for code in codes]
    print("\nPadded codes:")
    for i, (original, padded) in enumerate(zip(codes, padded_codes)):
        print(f"{original} → {padded}")


# ====================================================================================================
# str.lower()
# ====================================================================================================
def demo_lower():
    """
    str.lower(): Returns a copy of the string with all characters converted to lowercase.
    
    Syntax:
        string.lower()
    
    Returns:
        str: A new string with all characters converted to lowercase
    
    Notes:
        - Only cased characters (letters) are converted.
        - Numbers, symbols, and other characters remain unchanged.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) for the new string
    """
    # Basic example
    text = "Hello World"
    lowercase = text.lower()
    print(f"Original: '{text}' → Lowercase: '{lowercase}'")  # Output: Original: 'Hello World' → Lowercase: 'hello world'
    
    # Mixed case with numbers and symbols
    text2 = "HeLLo 123!@#"
    print(f"Original: '{text2}' → Lowercase: '{text2.lower()}'")  # Output: Original: 'HeLLo 123!@#' → Lowercase: 'hello 123!@#'
    
    # Already lowercase
    text3 = "already lowercase"
    print(f"Original: '{text3}' → Lowercase: '{text3.lower()}'")  # Output: Original: 'already lowercase' → Lowercase: 'already lowercase'
    
    # All uppercase
    text4 = "ALL UPPERCASE"
    print(f"Original: '{text4}' → Lowercase: '{text4.lower()}'")  # Output: Original: 'ALL UPPERCASE' → Lowercase: 'all uppercase'
    
    # Empty string
    empty = ""
    print(f"Empty string: '{empty.lower()}'")  # Output: Empty string: ''
    
    # Unicode support
    unicode_text = "Café RÉSUMÉ"
    print(f"Unicode: '{unicode_text}' → Lowercase: '{unicode_text.lower()}'")  # Output: Unicode: 'Café RÉSUMÉ' → Lowercase: 'café résumé'
    
    # Special Unicode case: German ß (already lowercase) vs SS
    german1 = "STRASSE"
    german2 = "STRAßE"
    print(f"German 'STRASSE': '{german1}' → Lowercase: '{german1.lower()}'")  # Output: German 'STRASSE': 'STRASSE' → Lowercase: 'strasse'
    print(f"German 'STRAßE': '{german2}' → Lowercase: '{german2.lower()}'")  # Output: German 'STRAßE': 'STRAßE' → Lowercase: 'straße'
    
    # Important: For true case-insensitive comparison, use casefold() instead of lower()
    print(f"Using lower(): 'STRASSE'.lower() == 'STRAßE'.lower(): {'STRASSE'.lower() == 'STRAßE'.lower()}")  # Output: False
    print(f"Using casefold(): 'STRASSE'.casefold() == 'STRAßE'.casefold(): {'STRASSE'.casefold() == 'STRAßE'.casefold()}")  # Output: True
    
    # Practical examples
    # 1. Case-insensitive comparison
    def case_insensitive_equal(str1, str2):
        return str1.lower() == str2.lower()
    
    word1 = "Python"
    word2 = "PYTHON"
    print(f"Are '{word1}' and '{word2}' equal (case-insensitive)? {case_insensitive_equal(word1, word2)}")  # Output: True
    
    # 2. Standardizing user input
    user_input = "YES"
    if user_input.lower() == "yes":
        print("User confirmed")
    else:
        print("User declined")
    
    # 3. Creating a case-insensitive dictionary
    class CaseInsensitiveDict(dict):
        def __getitem__(self, key):
            return super().__getitem__(key.lower())
        
        def __setitem__(self, key, value):
            super().__setitem__(key.lower(), value)
        
        def __contains__(self, key):
            return super().__contains__(key.lower())
    
    ci_dict = CaseInsensitiveDict()
    ci_dict["Key"] = "Value"
    print(f"Access with 'KEY': {ci_dict['KEY']}")  # Output: Access with 'KEY': Value
    print(f"Access with 'key': {ci_dict['key']}")  # Output: Access with 'key': Value
    print(f"'kEy' in dictionary: {'kEy' in ci_dict}")  # Output: 'kEy' in dictionary: True
    
    # 4. Normalizing text for search or storage
    def normalize_text(text):
        return text.lower().strip()
    
    search_terms = ["Python", "PYTHON ", " python"]
    normalized = [normalize_text(term) for term in search_terms]
    print(f"Original terms: {search_terms}")
    print(f"Normalized terms: {normalized}")  # All will be 'python'


# ====================================================================================================
# str.lstrip()
# ====================================================================================================
def demo_lstrip():
    """
    str.lstrip([chars]): Returns a copy of the string with leading characters removed.
    
    Syntax:
        string.lstrip([chars])
    
    Parameters:
        chars (str, optional): A string specifying the set of characters to remove. 
                              Defaults to removing whitespace.
    
    Returns:
        str: A new string with the leading characters removed
    
    Notes:
        - If chars is provided, all combinations of characters in chars will be removed from the left side.
        - The method stops when it encounters a character not in chars.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) for the new string
    """
    # Basic example - removing leading whitespace
    text = "   Hello, World!   "
    stripped = text.lstrip()
    print(f"Original: '{text}' → Left stripped: '{stripped}'")  # Output: Original: '   Hello, World!   ' → Left stripped: 'Hello, World!   '
    
    # Removing specific characters
    text2 = "...Python..."
    stripped2 = text2.lstrip('.')
    print(f"Remove leading dots: '{text2}' → '{stripped2}'")  # Output: Remove leading dots: '...Python...' → 'Python...'
    
    # Removing multiple specific characters
    text3 = "///---Python---///"
    stripped3 = text3.lstrip('/-')
    print(f"Remove leading slashes and hyphens: '{text3}' → '{stripped3}'")  # Output: Remove leading slashes and hyphens: '///---Python---///' → 'Python---///'
    
    # When no matching characters are found
    text4 = "Python"
    stripped4 = text4.lstrip('xyz')
    print(f"No matching chars: '{text4}' → '{stripped4}'")  # Output: No matching chars: 'Python' → 'Python'
    
    # Empty string
    empty = ""
    print(f"Empty string: '{empty.lstrip()}'")  # Output: Empty string: ''
    
    # Unicode support
    unicode_text = "　　こんにちは　　"  # Japanese text with ideographic spaces
    print(f"Unicode: '{unicode_text}' → '{unicode_text.lstrip()}'")  # Output: Unicode: '　　こんにちは　　' → 'こんにちは　　'
    
    # All types of whitespace
    whitespace_text = "\t\n\r\f\v Hello"
    print(f"All whitespace types: '{whitespace_text}' → '{whitespace_text.lstrip()}'")  # Removes all leading whitespace
    
    # Character order doesn't matter
    text5 = "abcdef"
    stripped5a = text5.lstrip('abc')
    stripped5b = text5.lstrip('cba')
    print(f"Character order in parameter: '{stripped5a}' vs '{stripped5b}'")  # Output: Character order in parameter: 'def' vs 'def'
    
    # Practical examples
    # 1. Removing common prefixes
    filename = "prefix_document.txt"
    clean_name = filename.lstrip("prefix_")
    print(f"Original filename: '{filename}' → Clean name: '{clean_name}'")  # Output: Original filename: 'prefix_document.txt' → Clean name: 'document.txt'
    
    # 2. Cleaning user input
    user_input = "    user response"
    cleaned = user_input.lstrip()
    print(f"User input: '{user_input}' → Cleaned: '{cleaned}'")  # Output: User input: '    user response' → Cleaned: 'user response'
    
    # 3. Removing markup or formatting characters
    markdown_text = "### Heading"
    plain_text = markdown_text.lstrip('# ')
    print(f"Markdown: '{markdown_text}' → Plain text: '{plain_text}'")  # Output: Markdown: '### Heading' → Plain text: 'Heading'
    
    # 4. Processing CSV data with potential leading whitespace
    csv_value = "  ,data with leading space"
    processed = csv_value.lstrip()
    print(f"CSV value: '{csv_value}' → Processed: '{processed}'")  # Output: CSV value: '  ,data with leading space' → Processed: ',data with leading space'
    
    # Important difference from strip() - only removes from left side
    text6 = "   spaced   "
    print(f"lstrip: '{text6.lstrip()}'")  # Output: lstrip: 'spaced   '
    print(f"strip: '{text6.strip()}'")    # Output: strip: 'spaced'


# ====================================================================================================
# str.maketrans()
# ====================================================================================================
def demo_maketrans():
    """
    str.maketrans(x[, y[, z]]): Returns a translation table for use with translate().
    
    Syntax:
        str.maketrans(x[, y[, z]])
    
    Parameters (has multiple forms):
        Form 1: str.maketrans(dict)
            dict: A dictionary mapping Unicode ordinals (integers) or characters to their replacements
        
        Form 2: str.maketrans(str1, str2)
            str1: String of characters to replace
            str2: String of replacement characters (must be same length as str1)
        
        Form 3: str.maketrans(str1, str2, str3)
            str1, str2: Same as Form 2
            str3: String of characters to delete
    
    Returns:
        dict: A translation mapping (dictionary) that can be used with translate()
    
    Raises:
        ValueError: If str1 and str2 have different lengths in Form 2
    
    Time Complexity: O(n) where n is the size of the translation table
    Space Complexity: O(n) for the translation table
    """
    # Form 1: Using a dictionary mapping
    # Replace 'a' with 'A', 'b' with 'B', etc.
    trans_dict = {'a': 'A', 'b': 'B', 'c': 'C'}
    trans_map1 = str.maketrans(trans_dict)
    
    text1 = "abcdef"
    translated1 = text1.translate(trans_map1)
    print(f"Using dictionary: '{text1}' → '{translated1}'")  # Output: Using dictionary: 'abcdef' → 'ABCdef'
    
    # Form 2: Using two strings of equal length
    # Replace 'a' with '1', 'e' with '2', 'i' with '3', etc.
    from_str = "aeiou"
    to_str = "12345"
    trans_map2 = str.maketrans(from_str, to_str)
    
    text2 = "Hello, World!"
    translated2 = text2.translate(trans_map2)
    print(f"Using two strings: '{text2}' → '{translated2}'")  # Output: Using two strings: 'Hello, World!' → 'H2ll4, W4rld!'
    
    # Form 3: Using two strings and a deletion string
    # Replace 'a' with '1', 'e' with '2', and delete spaces and commas
    delete_str = " ,"
    trans_map3 = str.maketrans(from_str, to_str, delete_str)
    
    text3 = "Hello, World!"
    translated3 = text3.translate(trans_map3)
    print(f"With deletions: '{text3}' → '{translated3}'")  # Output: With deletions: 'Hello, World!' → 'H2ll4W4rld!'
    
    # Using ordinals (integers) in the dictionary
    # Replace 'a' (97) with 'A' (65)
    trans_ord = {97: 65, 98: 66, 99: 67}  # a->A, b->B, c->C
    trans_map4 = str.maketrans(trans_ord)
    
    text4 = "abcdef"
    translated4 = text4.translate(trans_map4)
    print(f"Using ordinals: '{text4}' → '{translated4}'")  # Output: Using ordinals: 'abcdef' → 'ABCdef'
    
    # Error case: mismatched string lengths
    try:
        str.maketrans("abc", "xy")  # Different lengths
    except ValueError as e:
        print(f"Error with mismatched lengths: {e}")
    
    # Practical examples
    # 1. ROT13 cipher (rotate each letter by 13 positions)
    def rot13(text):
        lowercase = "abcdefghijklmnopqrstuvwxyz"
        uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Create a mapping that rotates each letter by 13 positions
        rot_lowercase = lowercase[13:] + lowercase[:13]
        rot_uppercase = uppercase[13:] + uppercase[:13]
        
        trans = str.maketrans(lowercase + uppercase, rot_lowercase + rot_uppercase)
        return text.translate(trans)
    
    message = "Hello, World!"
    encoded = rot13(message)
    decoded = rot13(encoded)  # ROT13 twice gets back the original
    
    print(f"\nROT13 Cipher:")
    print(f"Original: '{message}'")
    print(f"Encoded: '{encoded}'")
    print(f"Decoded: '{decoded}'")
    
    # 2. Remove vowels and replace spaces with hyphens
    def remove_vowels_and_format(text):
        vowels = "aeiouAEIOU"
        spaces = " "
        hyphens = "-"
        
        trans = str.maketrans("", "", vowels)  # Delete vowels
        text = text.translate(trans)
        
        trans2 = str.maketrans(spaces, hyphens)  # Replace spaces with hyphens
        return text.translate(trans2)
    
    original = "This is an example sentence"
    formatted = remove_vowels_and_format(original)
    print(f"\nFormatting:")
    print(f"Original: '{original}'")
    print(f"Formatted: '{formatted}'")
    
    # 3. Convert between ASCII and l33t speak
    def to_leetspeak(text):
        leet_map = {
            'a': '4', 'e': '3', 'i': '1', 'l': '1',
            'o': '0', 's': '5', 't': '7', 'A': '4',
            'E': '3', 'I': '1', 'L': '1', 'O': '0',
            'S': '5', 'T': '7'
        }
        trans = str.maketrans(leet_map)
        return text.translate(trans)
    
    normal_text = "Elite Hackers"
    leet_text = to_leetspeak(normal_text)
    print(f"\nL33t Speak:")
    print(f"Normal: '{normal_text}'")
    print(f"L33t: '{leet_text}'")


# ====================================================================================================
# str.partition()
# ====================================================================================================
def demo_partition():
    """
    str.partition(sep): Splits the string at the first occurrence of the separator.
    
    Syntax:
        string.partition(sep)
    
    Parameters:
        sep (str): The separator to split the string on
    
    Returns:
        tuple: A 3-tuple containing (part_before_separator, separator, part_after_separator).
              If the separator is not found, returns (original_string, '', '').
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) for the result tuple
    """
    # Basic example
    text = "Hello, World!"
    result = text.partition(", ")
    print(f"Partition '{text}' at ', ': {result}")  # Output: Partition 'Hello, World!' at ', ': ('Hello', ', ', 'World!')
    
    # Separator at the beginning
    text2 = "Hello World"
    result2 = text2.partition("H")
    print(f"Separator at beginning: {result2}")  # Output: Separator at beginning: ('', 'H', 'ello World')
    
    # Separator at the end
    text3 = "Hello World"
    result3 = text3.partition("d")
    print(f"Separator at end: {result3}")  # Output: Separator at end: ('Hello Worl', 'd', '')
    
    # Separator not found
    text4 = "Hello World"
    result4 = text4.partition("xyz")
    print(f"Separator not found: {result4}")  # Output: Separator not found: ('Hello World', '', '')
    
    # Empty separator (raises ValueError)
    try:
        text.partition("")
    except ValueError as e:
        print(f"Empty separator error: {e}")  # Output: Empty separator error: empty separator
    
    # Empty string
    empty = ""
    result5 = empty.partition("x")
    print(f"Empty string: {result5}")  # Output: Empty string: ('', '', '')
    
    # Multiple occurrences of separator (only splits at first occurrence)
    text6 = "apple,orange,banana"
    result6 = text6.partition(",")
    print(f"Multiple separators: {result6}")  # Output: Multiple separators: ('apple', ',', 'orange,banana')
    
    # Practical examples
    # 1. Extracting domain name from email
    def get_domain(email):
        _, _, domain = email.partition("@")
        return domain
    
    email = "user@example.com"
    domain = get_domain(email)
    print(f"\nDomain from '{email}': '{domain}'")  # Output: Domain from 'user@example.com': 'example.com'
    
    # 2. Parsing key-value pairs
    def parse_keyvalue(line):
        key, sep, value = line.partition("=")
        if sep:  # If separator was found
            return key.strip(), value.strip()
        return line.strip(), None
    
    config_lines = [
        "server=example.com",
        "port=8080",
        "comment line"
    ]
    
    print("\nParsing key-value pairs:")
    for line in config_lines:
        key, value = parse_keyvalue(line)
        if value:
            print(f"Key: '{key}', Value: '{value}'")
        else:
            print(f"No key-value pair: '{key}'")
    
    # 3. Splitting path into directory and filename
    def split_path(path):
        directory, _, filename = path.rpartition("/")  # Using rpartition to find last "/"
        return directory or ".", filename  # Default directory to "." if not found
    
    paths = [
        "/home/user/documents/file.txt",
        "file.txt",
        "folder/file.txt"
    ]
    
    print("\nSplitting paths:")
    for path in paths:
        directory, filename = split_path(path)
        print(f"Path: '{path}' → Directory: '{directory}', Filename: '{filename}'")
    
    # 4. Extracting protocol from URL
    def get_protocol(url):
        protocol, _, rest = url.partition("://")
        if "://" in url:
            return protocol
        return "unknown"
    
    urls = [
        "https://example.com",
        "ftp://files.example.com",
        "example.com"
    ]
    
    print("\nExtracting protocols:")
    for url in urls:
        protocol = get_protocol(url)
        print(f"URL: '{url}' → Protocol: '{protocol}'")


# ====================================================================================================
# str.removeprefix()
# ====================================================================================================
def demo_removeprefix():
    """
    str.removeprefix(prefix): Returns a string with the specified prefix removed if present.
    
    Syntax:
        string.removeprefix(prefix)
    
    Parameters:
        prefix (str): The prefix to remove from the string
    
    Returns:
        str: A copy of the string with the prefix removed if it exists, otherwise the original string
    
    Notes:
        - This method was added in Python 3.9
        - If the string doesn't start with the prefix, it returns the original string unchanged
    
    Time Complexity: O(n) where n is the length of the prefix
    Space Complexity: O(n) where n is the length of the result string
    """
    # Basic example
    text = "prefix_filename.txt"
    result = text.removeprefix("prefix_")
    print(f"Original: '{text}' → Remove prefix 'prefix_': '{result}'")  # Output: Original: 'prefix_filename.txt' → Remove prefix 'prefix_': 'filename.txt'
    
    # Prefix not found
    text2 = "filename.txt"
    result2 = text2.removeprefix("prefix_")
    print(f"Prefix not found: '{text2}' → '{result2}'")  # Output: Prefix not found: 'filename.txt' → 'filename.txt'
    
    # Case sensitivity
    text3 = "PREFIX_filename.txt"
    result3 = text3.removeprefix("prefix_")
    print(f"Case sensitive: '{text3}' → '{result3}'")  # Output: Case sensitive: 'PREFIX_filename.txt' → 'PREFIX_filename.txt'
    
    # Case insensitive approach
    def remove_prefix_case_insensitive(s, prefix):
        if s.lower().startswith(prefix.lower()):
            return s[len(prefix):]
        return s
    
    result3b = remove_prefix_case_insensitive(text3, "prefix_")
    print(f"Case insensitive: '{text3}' → '{result3b}'")  # Output: Case insensitive: 'PREFIX_filename.txt' → 'filename.txt'
    
    # Empty prefix
    text4 = "Hello"
    result4 = text4.removeprefix("")
    print(f"Empty prefix: '{text4}' → '{result4}'")  # Output: Empty prefix: 'Hello' → 'Hello'
    
    # Empty string
    empty = ""
    result5 = empty.removeprefix("prefix")
    print(f"Empty string: '{empty}' → '{result5}'")  # Output: Empty string: '' → ''
    
    # Prefix is the entire string
    text6 = "complete"
    result6 = text6.removeprefix("complete")
    print(f"Entire string: '{text6}' → '{result6}'")  # Output: Entire string: 'complete' → ''
    
    # Unicode support
    unicode_text = "👋Hello"
    result7 = unicode_text.removeprefix("👋")
    print(f"Unicode: '{unicode_text}' → '{result7}'")  # Output: Unicode: '👋Hello' → 'Hello'
    
    # Practical examples
    # 1. Cleaning up filenames
    filenames = [
        "temp_report.pdf",
        "temp_image.jpg",
        "document.docx",
        "temp_data.csv"
    ]
    
    print("\nCleaning filenames:")
    for filename in filenames:
        clean_name = filename.removeprefix("temp_")
        print(f"Original: '{filename}' → Cleaned: '{clean_name}'")
    
    # 2. Standardizing URLs
    def standardize_url(url):
        # Remove http/https prefix
        url = url.removeprefix("http://").removeprefix("https://")
        # Remove www. prefix
        url = url.removeprefix("www.")
        return url
    
    urls = [
        "https://www.example.com",
        "http://example.com",
        "www.example.org",
        "example.net"
    ]
    
    print("\nStandardizing URLs:")
    for url in urls:
        standardized = standardize_url(url)
        print(f"Original: '{url}' → Standardized: '{standardized}'")
    
    # 3. Processing command-line arguments
    def process_argument(arg):
        # Handle both --option and -o style arguments
        if arg.startswith("--"):
            return arg.removeprefix("--"), "long"
        elif arg.startswith("-"):
            return arg.removeprefix("-"), "short"
        return arg, "positional"
    
    args = ["--verbose", "-v", "filename.txt", "--output=file.out"]
    
    print("\nProcessing arguments:")
    for arg in args:
        name, arg_type = process_argument(arg)
        print(f"Argument: '{arg}' → Name: '{name}', Type: '{arg_type}'")
    
    # 4. Comparison with string slicing
    prefix = "prefix_"
    text = "prefix_name"
    
    # Traditional way (before Python 3.9)
    if text.startswith(prefix):
        result_old = text[len(prefix):]
    else:
        result_old = text
    
    # Using removeprefix
    result_new = text.removeprefix(prefix)
    
    print(f"\nTraditional slicing: '{result_old}'")
    print(f"Using removeprefix: '{result_new}'")


# ====================================================================================================
# str.removesuffix()
# ====================================================================================================
def demo_removesuffix():
    """
    str.removesuffix(suffix): Returns a string with the specified suffix removed if present.
    
    Syntax:
        string.removesuffix(suffix)
    
    Parameters:
        suffix (str): The suffix to remove from the string
    
    Returns:
        str: A copy of the string with the suffix removed if it exists, otherwise the original string
    
    Notes:
        - This method was added in Python 3.9
        - If the string doesn't end with the suffix, it returns the original string unchanged
    
    Time Complexity: O(n) where n is the length of the suffix
    Space Complexity: O(n) where n is the length of the result string
    """
    # Basic example
    text = "filename.txt"
    result = text.removesuffix(".txt")
    print(f"Original: '{text}' → Remove suffix '.txt': '{result}'")  # Output: Original: 'filename.txt' → Remove suffix '.txt': 'filename'
    
    # Suffix not found
    text2 = "filename.pdf"
    result2 = text2.removesuffix(".txt")
    print(f"Suffix not found: '{text2}' → '{result2}'")  # Output: Suffix not found: 'filename.pdf' → 'filename.pdf'
    
    # Case sensitivity
    text3 = "filename.TXT"
    result3 = text3.removesuffix(".txt")
    print(f"Case sensitive: '{text3}' → '{result3}'")  # Output: Case sensitive: 'filename.TXT' → 'filename.TXT'
    
    # Case insensitive approach
    def remove_suffix_case_insensitive(s, suffix):
        if s.lower().endswith(suffix.lower()):
            return s[:-len(suffix)]
        return s
    
    result3b = remove_suffix_case_insensitive(text3, ".txt")
    print(f"Case insensitive: '{text3}' → '{result3b}'")  # Output: Case insensitive: 'filename.TXT' → 'filename'
    
    # Empty suffix
    text4 = "Hello"
    result4 = text4.removesuffix("")
    print(f"Empty suffix: '{text4}' → '{result4}'")  # Output: Empty suffix: 'Hello' → 'Hello'
    
    # Empty string
    empty = ""
    result5 = empty.removesuffix("suffix")
    print(f"Empty string: '{empty}' → '{result5}'")  # Output: Empty string: '' → ''
    
    # Suffix is the entire string
    text6 = "complete"
    result6 = text6.removesuffix("complete")
    print(f"Entire string: '{text6}' → '{result6}'")  # Output: Entire string: 'complete' → ''
    
    # Unicode support
    unicode_text = "Hello👋"
    result7 = unicode_text.removesuffix("👋")
    print(f"Unicode: '{unicode_text}' → '{result7}'")  # Output: Unicode: 'Hello👋' → 'Hello'
    
    # Practical examples
    # 1. Removing file extensions
    def get_filename_without_extension(filename):
        return filename.removesuffix(".txt").removesuffix(".pdf").removesuffix(".docx")
    
    filenames = [
        "report.txt",
        "image.pdf",
        "document.docx",
        "data"
    ]
    
    print("\nRemoving file extensions:")
    for filename in filenames:
        base_name = get_filename_without_extension(filename)
        print(f"Original: '{filename}' → Base name: '{base_name}'")
    
    # 2. Better approach for removing file extensions
    import os
    
    def get_filename_without_extension_better(filename):
        root, _ = os.path.splitext(filename)
        return root
    
    print("\nBetter file extension removal:")
    for filename in filenames:
        base_name = get_filename_without_extension_better(filename)
        print(f"Original: '{filename}' → Base name: '{base_name}'")
    
    # 3. Cleaning up URLs
    def clean_url(url):
        # Remove trailing slash
        return url.removesuffix("/")
    
    urls = [
        "https://example.com/",
        "https://example.org/path/",
        "https://example.net"
    ]
    
    print("\nCleaning URLs:")
    for url in urls:
        cleaned = clean_url(url)
        print(f"Original: '{url}' → Cleaned: '{cleaned}'")
    
    # 4. Removing common text markers
    def remove_markers(text):
        return text.removesuffix(" (edited)").removesuffix(" [DRAFT]").removesuffix(" *")
    
    texts = [
        "Document Title (edited)",
        "Report [DRAFT]",
        "Note *",
        "Plain Text"
    ]
    
    print("\nRemoving text markers:")
    for text in texts:
        clean_text = remove_markers(text)
        print(f"Original: '{text}' → Cleaned: '{clean_text}'")
    
    # 5. Comparison with string slicing
    suffix = ".txt"
    text = "filename.txt"
    
    # Traditional way (before Python 3.9)
    if text.endswith(suffix):
        result_old = text[:-len(suffix)]
    else:
        result_old = text
    
    # Using removesuffix
    result_new = text.removesuffix(suffix)
    
    print(f"\nTraditional slicing: '{result_old}'")
    print(f"Using removesuffix: '{result_new}'")


# ====================================================================================================
# str.replace()
# ====================================================================================================
def demo_replace():
    """
    str.replace(old, new[, count]): Returns a copy of the string with all occurrences of substring old
                                     replaced by new.
    
    Syntax:
        string.replace(old, new[, count])
    
    Parameters:
        old (str): The substring to replace
        new (str): The replacement string
        count (int, optional): Maximum number of occurrences to replace. Defaults to all occurrences.
    
    Returns:
        str: A copy of the string with replacements made
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) for the new string
    """
    # Basic example
    text = "Hello, world! Hello, universe!"
    replaced = text.replace("Hello", "Hi")
    print(f"Original: '{text}' → Replace 'Hello' with 'Hi': '{replaced}'")  # Output: Original: 'Hello, world! Hello, universe!' → Replace 'Hello' with 'Hi': 'Hi, world! Hi, universe!'
    
    # Using count parameter to limit replacements
    replaced_count = text.replace("Hello", "Hi", 1)
    print(f"Replace with count=1: '{replaced_count}'")  # Output: Replace with count=1: 'Hi, world! Hello, universe!'
    
    # Replace with empty string (removing)
    text2 = "Remove all spaces"
    removed = text2.replace(" ", "")
    print(f"Remove spaces: '{text2}' → '{removed}'")  # Output: Remove spaces: 'Remove all spaces' → 'Removeallspaces'
    
    # Replace when substring is not found
    text3 = "No changes"
    not_found = text3.replace("xyz", "abc")
    print(f"Substring not found: '{text3}' → '{not_found}'")  # Output: Substring not found: 'No changes' → 'No changes'
    
    # Case sensitivity
    text4 = "Case CASE case"
    case_sensitive = text4.replace("case", "example")
    print(f"Case sensitive: '{text4}' → '{case_sensitive}'")  # Output: Case sensitive: 'Case CASE case' → 'Case CASE example'
    
    # Case insensitive replacement
    def replace_case_insensitive(text, old, new):
        # This is a basic implementation - re.sub with re.IGNORECASE would be more efficient
        result = ""
        i = 0
        while i < len(text):
            if text[i:i+len(old)].lower() == old.lower():
                result += new
                i += len(old)
            else:
                result += text[i]
                i += 1
        return result
    
    case_insensitive = replace_case_insensitive(text4, "case", "example")
    print(f"Case insensitive: '{text4}' → '{case_insensitive}'")  # Output: Case insensitive: 'Case CASE case' → 'example example example'
    
    # Using regex for more advanced replacement
    import re
    text5 = "Case CASE case"
    regex_replace = re.sub(r"case", "example", text5, flags=re.IGNORECASE)
    print(f"Using regex: '{text5}' → '{regex_replace}'")  # Output: Using regex: 'Case CASE case' → 'example example example'
    
    # Empty strings
    empty = ""
    print(f"Empty string: '{empty.replace('a', 'b')}'")  # Output: Empty string: ''
    
    # Replace with empty old string (raises ValueError)
    try:
        text.replace("", "x")
    except ValueError as e:
        print(f"Empty old string error: {e}")  # Output: Empty old string error: empty pattern string
    
    # Replacing with longer/shorter string
    text6 = "short to long"
    longer = text6.replace("short", "much longer")
    shorter = text6.replace("long", "")
    print(f"Replace with longer: '{text6}' → '{longer}'")  # Output: Replace with longer: 'short to long' → 'much longer to long'
    print(f"Replace with shorter: '{text6}' → '{shorter}'")  # Output: Replace with shorter: 'short to long' → 'short to '
    
    # Practical examples
    # 1. Censoring sensitive information
    def censor_email(text):
        import re
        # Find email pattern and replace with censored version
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, "[EMAIL REDACTED]", text)
    
    sensitive = "Contact me at user@example.com or call 555-1234"
    censored = censor_email(sensitive)
    print(f"\nCensored: '{censored}'")
    
    # 2. Normalizing line endings
    def normalize_newlines(text):
        # First convert Windows line endings to Unix
        text = text.replace("\r\n", "\n")
        # Then convert old Mac line endings to Unix
        return text.replace("\r", "\n")
    
    mixed_newlines = "Line 1\r\nLine 2\rLine 3\nLine 4"
    normalized = normalize_newlines(mixed_newlines)
    print(f"\nNormalized newlines:")
    print(f"Original (escaped): {repr(mixed_newlines)}")
    print(f"Normalized (escaped): {repr(normalized)}")
    
    # 3. Cleaning up user input
    def clean_user_input(text):
        # Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return cleaned.strip()
    
    user_input = "  This   has    many    spaces   "
    cleaned_input = clean_user_input(user_input)
    print(f"\nUser input: '{user_input}'")
    print(f"Cleaned: '{cleaned_input}'")
    
    # 4. Simple template system
    template = "Dear {name}, Your appointment is on {date} at {time}."
    filled = template.replace("{name}", "John").replace("{date}", "Monday").replace("{time}", "2:30 PM")
    print(f"\nTemplate filling:")
    print(f"Template: '{template}'")
    print(f"Filled: '{filled}'")


# ====================================================================================================
# str.rfind()
# ====================================================================================================
def demo_rfind():
    """
    str.rfind(sub[, start[, end]]): Returns the highest index where substring is found.
    
    Syntax:
        string.rfind(substring[, start[, end]])
    
    Parameters:
        sub (str): The substring to find
        start (int, optional): The starting index. Defaults to 0.
        end (int, optional): The ending index. Defaults to the end of the string.
    
    Returns:
        int: The highest index where substring is found, or -1 if not found
    
    Time Complexity: O(n*m) where n is the length of the string and m is the length of the substring
    Space Complexity: O(1)
    """
    # Basic example
    text = "Hello, world! Hello, universe!"
    index = text.rfind("Hello")
    print(f"Last occurrence of 'Hello' in '{text}': index {index}")  # Output: Last occurrence of 'Hello': index 14
    
    # Compare with find() (finds first occurrence)
    first_index = text.find("Hello")
    print(f"First occurrence (find): {first_index}, Last occurrence (rfind): {index}")  # Output: First occurrence (find): 0, Last occurrence (rfind): 14
    
    # Using start and end parameters
    text2 = "one two one two one two"
    index2 = text2.rfind("one", 0, 10)
    print(f"Last 'one' between indices 0 and 10: {index2}")  # Output: Last 'one' between indices 0 and 10: 8
    
    # Substring not found
    not_found = text.rfind("xyz")
    print(f"'xyz' in text: index {not_found}")  # Output: 'xyz' in text: index -1
    
    # Case sensitivity
    case_sensitive = text.rfind("hello")
    print(f"'hello' (lowercase) in text: index {case_sensitive}")  # Output: 'hello' (lowercase) in text: index -1
    
    # Empty string
    empty_string = text.rfind("")
    print(f"Empty string in text: index {empty_string}")  # Output: Empty string in text: index 29 (length of the string)
    
    # Practical examples
    # 1. Finding file extension
    def get_file_extension(filename):
        dot_pos = filename.rfind(".")
        if dot_pos != -1:
            return filename[dot_pos:]
        return ""
    
    filenames = ["document.txt", "image.png", "script.py", "noextension"]
    
    print("\nFile extensions:")
    for filename in filenames:
        extension = get_file_extension(filename)
        print(f"Filename: '{filename}' → Extension: '{extension}'")
    
    # 2. Finding last directory separator in a path
    def split_path(path):
        separator_pos = path.rfind("/")
        if separator_pos != -1:
            return path[:separator_pos], path[separator_pos+1:]
        return "", path
    
    paths = [
        "/home/user/documents/file.txt",
        "file.txt",
        "folder/file.txt"
    ]
    
    print("\nSplitting paths:")
    for path in paths:
        directory, filename = split_path(path)
        print(f"Path: '{path}' → Directory: '{directory}', Filename: '{filename}'")
    
    # 3. Extracting domain from email address
    def get_domain(email):
        at_pos = email.rfind("@")
        if at_pos != -1:
            return email[at_pos+1:]
        return ""
    
    emails = ["user@example.com", "contact@company.co.uk", "invalid-email"]
    
    print("\nExtracting domains:")
    for email in emails:
        domain = get_domain(email)
        if domain:
            print(f"Email: '{email}' → Domain: '{domain}'")
        else:
            print(f"Email: '{email}' → Invalid format")
    
    # 4. Finding the last word in a sentence
    def get_last_word(sentence):
        # Remove trailing punctuation
        clean_sentence = sentence.rstrip(".!?,:;")
        space_pos = clean_sentence.rfind(" ")
        if space_pos != -1:
            return clean_sentence[space_pos+1:]
        return clean_sentence
    
    sentences = [
        "This is a test sentence.",
        "What is the last word?",
        "Single!"
    ]
    
    print("\nExtracting last words:")
    for sentence in sentences:
        last_word = get_last_word(sentence)
        print(f"Sentence: '{sentence}' → Last word: '{last_word}'")


# ====================================================================================================
# str.rindex()
# ====================================================================================================
def demo_rindex():
    """
    str.rindex(sub[, start[, end]]): Like rfind(), but raises ValueError when the substring is not found.
    
    Syntax:
        string.rindex(substring[, start[, end]])
    
    Parameters:
        sub (str): The substring to find
        start (int, optional): The starting index. Defaults to 0.
        end (int, optional): The ending index. Defaults to the end of the string.
    
    Returns:
        int: The highest index where substring is found
    
    Raises:
        ValueError: When the substring is not found
    
    Time Complexity: O(n*m) where n is the length of the string and m is the length of the substring
    Space Complexity: O(1)
    """
    # Basic example
    text = "Hello, world! Hello, universe!"
    index = text.rindex("Hello")
    print(f"Last occurrence of 'Hello' in '{text}': index {index}")  # Output: Last occurrence of 'Hello': index 14
    
    # Using start and end parameters
    text2 = "one two one two one two"
    index2 = text2.rindex("one", 0, 15)
    print(f"Last 'one' between indices 0 and 15: {index2}")  # Output depends on exact string
    
    # Compare with rfind()
    print("\nComparison with rfind():")
    print(f"rfind('Hello'): {text.rfind('Hello')}")  # Output: rfind('Hello'): 14
    print(f"rindex('Hello'): {text.rindex('Hello')}")  # Output: rindex('Hello'): 14
    
    # Handling substring not found
    print("\nHandling substring not found:")
    
    # Using rfind() (returns -1)
    not_found = text.rfind("xyz")
    print(f"rfind('xyz'): {not_found}")  # Output: rfind('xyz'): -1
    
    # Using rindex() (raises ValueError)
    try:
        text.rindex("xyz")
    except ValueError as e:
        print(f"rindex('xyz') raised: {e}")  # Output: rindex('xyz') raised: substring not found
    
    # Case sensitivity
    try:
        text.rindex("hello")  # Case-sensitive
    except ValueError as e:
        print(f"Case sensitivity: {e}")  # Output: Case sensitivity: substring not found
    
    # Empty string
    empty_index = text.rindex("")
    print(f"Index of empty string: {empty_index}")  # Output: Index of empty string: 29 (length of the string)
    
    # Empty string input
    try:
        "".rindex("a")
    except ValueError as e:
        print(f"Empty input error: {e}")  # Output: Empty input error: substring not found
    
    # Practical examples
    # 1. Safe extraction of file extension
    def get_extension_safe(filename):
        try:
            dot_index = filename.rindex(".")
            return filename[dot_index:]
        except ValueError:
            return ""  # No extension found
    
    filenames = ["document.txt", "image.jpg.png", "noextension"]
    
    print("\nSafe extension extraction:")
    for filename in filenames:
        extension = get_extension_safe(filename)
        print(f"Filename: '{filename}' → Extension: '{extension}'")
    
    # 2. Finding the last occurrence of a tag in HTML
    def find_last_tag(html, tag):
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        
        try:
            open_index = html.rindex(open_tag)
            close_index = html.rindex(close_tag)
            
            if open_index < close_index:
                content_start = open_index + len(open_tag)
                return html[content_start:close_index]
        except ValueError:
            return None
        
        return None
    
    html = "<p>First paragraph</p><p>Second paragraph</p>"
    last_p_content = find_last_tag(html, "p")
    
    print(f"\nLast paragraph content: '{last_p_content}'")
    
    # 3. Using try-except for robust code
    def extract_substring(text, start_marker, end_marker):
        try:
            start_index = text.rindex(start_marker) + len(start_marker)
            end_index = text.rindex(end_marker, start_index)
            return text[start_index:end_index]
        except ValueError:
            return None
    
    data = "START:First data END START:Second data END"
    result = extract_substring(data, "START:", " END")
    
    print(f"\nExtracted substring: '{result}'")
    
    # 4. Getting last element in a delimited string
    def get_last_element(text, delimiter):
        try:
            last_delim_index = text.rindex(delimiter)
            return text[last_delim_index + len(delimiter):]
        except ValueError:
            return text  # No delimiter found
    
    paths = [
        "a/b/c/d",
        "item1,item2,item3",
        "single"
    ]
    
    print("\nExtracting last elements:")
    for path in paths:
        delimiter = "/" if "/" in path else ","
        last = get_last_element(path, delimiter)
        print(f"From '{path}' with delimiter '{delimiter}': '{last}'")


# ====================================================================================================
# str.rjust()
# ====================================================================================================
def demo_rjust():
    """
    str.rjust(width[, fillchar]): Returns a right-justified string of specified width.
    
    Syntax:
        string.rjust(width[, fillchar])
    
    Parameters:
        width (int): The total width of the resulting string
        fillchar (str, optional): The character to pad with. Defaults to space.
    
    Returns:
        str: A new string that is right-justified within a field of the specified width
    
    Raises:
        TypeError: If width is not an integer, or fillchar is not a character
    
    Time Complexity: O(n) where n is the width
    Space Complexity: O(n) for the new string
    """
    # Basic example
    text = "Hello"
    right_justified = text.rjust(10)
    print(f"Original: '{text}' → Right justified (width=10): '{right_justified}'")  # Output: Original: 'Hello' → Right justified (width=10): '     Hello'
    
    # Using custom fill character
    right_justified_star = text.rjust(10, '*')
    print(f"Right justified with '*': '{right_justified_star}'")  # Output: Right justified with '*': '*****Hello'
    
    # When width is less than or equal to the length of the string
    print(f"Width equal to length: '{text.rjust(5)}'")  # Output: Width equal to length: 'Hello'
    print(f"Width less than length: '{text.rjust(3)}'")  # Output: Width less than length: 'Hello'
    
    # Various fill characters
    print("\nVarious fill characters:")
    fill_chars = [' ', '-', '.', '=']
    for char in fill_chars:
        print(f"'{text.rjust(10, char)}'")
    
    # Empty string
    empty = ""
    print(f"Empty string: '{empty.rjust(5, '_')}'")  # Output: Empty string: '_____'
    
    # Unicode/emoji support
    emoji = "🐍"
    print(f"Unicode: '{emoji.rjust(5, '.')}'")  # Output: Unicode: '....🐍'
    
    # Exception cases
    try:
        # TypeError: rjust() argument 1 must be int, not str
        text.rjust("10")
    except TypeError as e:
        print(f"Error with non-integer width: {e}")
    
    try:
        # TypeError: rjust() argument 2 must be str of length 1, not str
        text.rjust(10, "**")
    except TypeError as e:
        print(f"Error with multi-character fillchar: {e}")
    
    # Practical examples
    # 1. Creating a right-aligned number table
    numbers = [42, 8, 15, 16, 23, 4]
    
    print("\nRight-aligned number table:")
    for num in numbers:
        print(str(num).rjust(5))
    
    # 2. Formatting financial data
    def format_money(amount):
        return f"${str(amount):.2f}".rjust(10)
    
    transactions = [12.34, 56.78, 90.12]