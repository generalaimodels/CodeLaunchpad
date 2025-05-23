
#!/usr/bin/env python3
"""
Python Regular Expressions (re module) - Comprehensive Guide
===========================================================

This file contains a complete guide to Python's re module with executable examples.
Regular expressions are powerful patterns for string matching and manipulation.
"""
import re

# ==========================================
# SECTION 1: BASICS AND SIMPLE MATCHING
# ==========================================

def basic_matching():
    """Basic pattern matching examples."""
    # The match() function checks if the pattern matches at the BEGINNING of the string
    text = "Python is amazing"
    
    # Basic literal matching
    result = re.match(r"Python", text)
    print(f"1. match 'Python' at start: {result is not None}")  # True
    
    # match() only matches at the start of the string
    result = re.match(r"is", text)
    print(f"2. match 'is' at start: {result is not None}")  # False
    
    # search() finds pattern anywhere in the string
    result = re.search(r"is", text)
    print(f"3. search 'is' anywhere: {result is not None}")  # True
    
    # Using fullmatch() to match the entire string
    result = re.fullmatch(r"Python is amazing", text)
    print(f"4. fullmatch entire string: {result is not None}")  # True
    
    # Case sensitivity
    result = re.search(r"python", text)
    print(f"5. Case sensitive search: {result is not None}")  # False
    
    # Using case-insensitive flag
    result = re.search(r"python", text, re.IGNORECASE)
    print(f"6. Case insensitive search: {result is not None}")  # True


# ==========================================
# SECTION 2: METACHARACTERS & SPECIAL SEQUENCES
# ==========================================

def metacharacters():
    """Examples of metacharacters in regular expressions."""
    # . (dot) - Matches any character except newline
    result = re.search(r"P.thon", "Python")
    print(f"1. Dot matches any character: {result is not None}")  # True
    
    # ^ - Matches start of string
    result = re.search(r"^Python", "Python is great")
    print(f"2. ^ matches start of string: {result is not None}")  # True
    
    # $ - Matches end of string
    result = re.search(r"great$", "Python is great")
    print(f"3. $ matches end of string: {result is not None}")  # True
    
    # * - Matches 0 or more repetitions
    result = re.search(r"ab*c", "ac")  # b appears 0 times
    print(f"4. * matches 0 occurrences: {result is not None}")  # True
    
    result = re.search(r"ab*c", "abc")  # b appears 1 time
    print(f"5. * matches 1 occurrence: {result is not None}")  # True
    
    result = re.search(r"ab*c", "abbc")  # b appears 2 times
    print(f"6. * matches multiple occurrences: {result is not None}")  # True
    
    # + - Matches 1 or more repetitions
    result = re.search(r"ab+c", "ac")  # b appears 0 times
    print(f"7. + requires at least 1 occurrence: {result is not None}")  # False
    
    result = re.search(r"ab+c", "abbc")  # b appears 2 times
    print(f"8. + matches multiple occurrences: {result is not None}")  # True
    
    # ? - Matches 0 or 1 repetition
    result = re.search(r"ab?c", "ac")  # b appears 0 times
    print(f"9. ? matches 0 occurrences: {result is not None}")  # True
    
    result = re.search(r"ab?c", "abc")  # b appears 1 time
    print(f"10. ? matches 1 occurrence: {result is not None}")  # True
    
    result = re.search(r"ab?c", "abbc")  # b appears 2 times
    print(f"11. ? doesn't match multiple occurrences: {result is not None}")  # False
    
    # {} - Matches exact number of repetitions
    result = re.search(r"a{3}", "aaa")  # Exactly 3 a's
    print(f"12. {{n}} matches exact repetitions: {result is not None}")  # True
    
    # {m,n} - Matches from m to n repetitions
    result = re.search(r"a{2,4}", "aa")  # 2 a's (within range)
    print(f"13. {{m,n}} matches range of repetitions (lower): {result is not None}")  # True
    
    result = re.search(r"a{2,4}", "aaaa")  # 4 a's (within range)
    print(f"14. {{m,n}} matches range of repetitions (upper): {result is not None}")  # True
    
    result = re.search(r"a{2,4}", "aaaaa")  # Matches first 4 a's
    print(f"15. {{m,n}} matches within string: {result is not None}")  # True
    
    # | - Alternation (OR)
    result = re.search(r"cat|dog", "I have a cat")
    print(f"16. | matches alternatives (first): {result is not None}")  # True
    
    result = re.search(r"cat|dog", "I have a dog")
    print(f"17. | matches alternatives (second): {result is not None}")  # True
    
    # () - Grouping
    result = re.search(r"(ab)+", "ababab")  # Group 'ab' repeated
    print(f"18. () groups patterns: {result is not None}")  # True
    
    # Escaping metacharacters with \
    result = re.search(r"\.", "This is a sentence.")  # Literal dot
    print(f"19. Escaping metacharacters: {result is not None}")  # True


def character_classes():
    """Examples of character classes in regular expressions."""
    # [abc] - Matches any character in the set
    result = re.search(r"[aeiou]", "hello")  # Contains vowel
    print(f"1. Character set matches any included character: {result is not None}")  # True
    
    # [^abc] - Matches any character NOT in the set
    result = re.search(r"[^aeiou]", "hello")  # Contains non-vowel
    print(f"2. Negated set matches any excluded character: {result is not None}")  # True
    
    # [a-z] - Range of characters
    result = re.search(r"[a-z]", "HELLO123")  # Contains lowercase letter
    print(f"3. Character range (lowercase): {result is not None}")  # False
    
    # [A-Z] - Range of uppercase characters
    result = re.search(r"[A-Z]", "HELLO123")  # Contains uppercase letter
    print(f"4. Character range (uppercase): {result is not None}")  # True
    
    # [0-9] - Range of digits
    result = re.search(r"[0-9]", "HELLO123")  # Contains digit
    print(f"5. Character range (digits): {result is not None}")  # True
    
    # Combined ranges
    result = re.search(r"[a-zA-Z0-9]", "!@#$%^")  # Contains alphanumeric
    print(f"6. Combined character ranges: {result is not None}")  # False


def special_sequences():
    """Examples of special sequences in regular expressions."""
    # \d - Matches any digit (equivalent to [0-9])
    result = re.search(r"\d", "abc123")
    print(f"1. \\d matches digits: {result is not None}")  # True
    
    # \D - Matches any non-digit
    result = re.search(r"\D", "123")
    print(f"2. \\D matches non-digits: {result is not None}")  # False
    
    # \w - Matches word characters (alphanumeric + underscore)
    result = re.search(r"\w", "!@#$%^&")
    print(f"3. \\w matches word characters: {result is not None}")  # False
    
    # \W - Matches non-word characters
    result = re.search(r"\W", "abcdef")
    print(f"4. \\W matches non-word characters: {result is not None}")  # False
    
    # \s - Matches whitespace
    result = re.search(r"\s", "hello world")
    print(f"5. \\s matches whitespace: {result is not None}")  # True
    
    # \S - Matches non-whitespace
    result = re.search(r"\S", "   \n\t   ")
    print(f"6. \\S matches non-whitespace: {result is not None}")  # False
    
    # \b - Matches word boundary
    result = re.search(r"\bcat\b", "The cat sat")  # 'cat' as a whole word
    print(f"7. \\b matches word boundary: {result is not None}")  # True
    
    result = re.search(r"\bcat\b", "The category")  # 'cat' not a whole word
    print(f"8. \\b requires word boundary: {result is not None}")  # False
    
    # \B - Matches non-word boundary
    result = re.search(r"\Bcat\B", "The category")  # 'cat' inside a word
    print(f"9. \\B matches non-word boundary: {result is not None}")  # False
    
    result = re.search(r"\Bcat\B", "location")  # 'cat' inside a word
    print(f"10. \\B matches characters inside word: {result is not None}")  # True


# ==========================================
# SECTION 3: SEARCH, MATCH & REPLACE OPERATIONS
# ==========================================

def search_operations():
    """Different matching and searching operations."""
    text = "The quick brown fox jumps over the lazy dog"
    
    # re.search() - Find pattern anywhere in the string
    result = re.search(r"fox", text)
    if result:
        print(f"1. search 'fox': found at position {result.start()}-{result.end()}")
    
    # re.match() - Match pattern at the beginning of the string
    result = re.match(r"The", text)
    if result:
        print(f"2. match 'The': found at position {result.start()}-{result.end()}")
    
    result = re.match(r"quick", text)  # Not at the beginning
    print(f"3. match 'quick' at beginning: {result is not None}")  # False
    
    # re.fullmatch() - Match entire string
    result = re.fullmatch(r"The quick brown fox jumps over the lazy dog", text)
    print(f"4. fullmatch entire string: {result is not None}")  # True
    
    # re.findall() - Find all occurrences (returns list of matches)
    result = re.findall(r"the", text, re.IGNORECASE)
    print(f"5. findall 'the' (case insensitive): {result}")  # ['The', 'the']
    
    # re.finditer() - Find all occurrences (returns iterator of match objects)
    result = re.finditer(r"[a-z]{4,}", text)  # Words with 4+ lowercase letters
    positions = [(m.group(), m.start(), m.end()) for m in result]
    print(f"6. finditer words 4+ chars: {positions}")
    
    # Using span() to get match positions
    match = re.search(r"brown", text)
    if match:
        span = match.span()
        print(f"7. span() for 'brown': {span}")


def replace_operations():
    """String replacement operations with regular expressions."""
    text = "The date is 2023-03-15 and tomorrow is 2023-03-16."
    
    # re.sub() - Replace all occurrences
    # Convert dates from YYYY-MM-DD to MM/DD/YYYY
    result = re.sub(r"(\d{4})-(\d{2})-(\d{2})", r"\2/\3/\1", text)
    print(f"1. sub() date format conversion: {result}")
    
    # re.subn() - Replace all occurrences and count replacements
    result, count = re.subn(r"\d", "X", text)
    print(f"2. subn() replace digits: {result}, {count} replacements")
    
    # Replacement with a function
    def double_digits(match):
        """Double any digit found."""
        digit = match.group(0)
        return str(int(digit) * 2)
    
    result = re.sub(r"\d", double_digits, "The number is 7")
    print(f"3. sub() with function: {result}")
    
    # Limiting replacements with count parameter
    text = "one two three four five"
    result = re.sub(r"\w+", "WORD", text, count=3)
    print(f"4. sub() with count limit: {result}")


# ==========================================
# SECTION 4: GROUPS & CAPTURING
# ==========================================

def groups_and_capturing():
    """Working with groups and capturing in regular expressions."""
    # Basic capturing group
    text = "John Smith"
    match = re.search(r"(\w+) (\w+)", text)
    if match:
        print(f"1. Capturing groups: First={match.group(1)}, Last={match.group(2)}")
    
    # Accessing all groups
    print(f"2. All groups: {match.groups()}")
    
    # Named groups
    match = re.search(r"(?P<first>\w+) (?P<last>\w+)", text)
    if match:
        print(f"3. Named groups: First={match.group('first')}, Last={match.group('last')}")
        print(f"4. Named groups dict: {match.groupdict()}")
    
    # Non-capturing groups (?:...)
    text = "The color is #FF0000 and the code is 123"
    matches = re.findall(r"(?:color|code) is (\S+)", text)
    print(f"5. Non-capturing groups: {matches}")  # Captures what comes after, not 'color'/'code'
    
    # Backreferences - using \1, \2, etc. to refer to previous groups
    text = "hello hello world"
    match = re.search(r"(\w+) \1", text)  # Find repeated words
    if match:
        print(f"6. Backreference finds: {match.group(0)}")
    
    # Named backreferences
    html = "<div>Content</div>"
    match = re.search(r"<(?P<tag>\w+)>.*?</(?P=tag)>", html)
    if match:
        print(f"7. Named backreference finds: {match.group(0)}")


# ==========================================
# SECTION 5: LOOKAHEAD & LOOKBEHIND ASSERTIONS
# ==========================================

def lookaround_assertions():
    """Examples of lookahead and lookbehind assertions."""
    # Positive lookahead (?=...) - Match only if followed by pattern
    text = "The price is $50 and €30"
    matches = re.findall(r"\d+(?=\s*€)", text)  # Find numbers followed by €
    print(f"1. Positive lookahead (euros): {matches}")
    
    # Negative lookahead (?!...) - Match only if NOT followed by pattern
    matches = re.findall(r"\d+(?!\s*€)", text)  # Find numbers NOT followed by €
    print(f"2. Negative lookahead (not euros): {matches}")
    
    # Positive lookbehind (?<=...) - Match only if preceded by pattern
    matches = re.findall(r"(?<=\$)\d+", text)  # Find numbers preceded by $
    print(f"3. Positive lookbehind (dollars): {matches}")
    
    # Negative lookbehind (?<!...) - Match only if NOT preceded by pattern
    matches = re.findall(r"(?<!\$)\d+", text)  # Find numbers NOT preceded by $
    print(f"4. Negative lookbehind (not dollars): {matches}")
    
    # Combined lookarounds
    text = "Product codes: ABC123, XYZ456, DEF-789"
    # Find product codes with letters followed by numbers, without hyphens
    matches = re.findall(r"(?<![A-Z0-9]-)[A-Z]+\d+", text)
    print(f"5. Combined lookarounds (valid product codes): {matches}")


# ==========================================
# SECTION 6: REGEX FLAGS
# ==========================================

def regex_flags():
    """Examples of different regex flags."""
    text = """This is line one.
    This is line TWO.
    This is line three."""
    
    # re.IGNORECASE (re.I) - Case insensitive matching
    matches = re.findall(r"this", text, re.IGNORECASE)
    print(f"1. IGNORECASE flag: {matches}")  # Finds all 'this' regardless of case
    
    # re.MULTILINE (re.M) - ^ and $ match at beginning/end of each line
    matches = re.findall(r"^This", text, re.MULTILINE)
    print(f"2. MULTILINE flag: {matches}")  # Finds 'This' at start of each line
    
    # re.DOTALL (re.S) - Dot matches newlines too
    match_default = re.search(r"one.*three", text) is not None
    match_dotall = re.search(r"one.*three", text, re.DOTALL) is not None
    print(f"3. Without DOTALL: {match_default}, With DOTALL: {match_dotall}")
    
    # re.VERBOSE (re.X) - Allow comments and whitespace in pattern
    phone_pattern = re.compile(r"""
        \(?\d{3}\)?  # Area code, optional parentheses
        [-\s]?       # Optional separator (hyphen or space)
        \d{3}        # Exchange code
        [-\s]?       # Optional separator
        \d{4}        # Station number
        """, re.VERBOSE)
    
    match = phone_pattern.search("Call (123) 456-7890 today!")
    if match:
        print(f"4. VERBOSE flag phone match: {match.group()}")
    
    # Combining flags with | (bitwise OR)
    matches = re.findall(r"^this", text, re.IGNORECASE | re.MULTILINE)
    print(f"5. Combined flags: {matches}")
    
    # re.ASCII (re.A) - Make \w, \W, \b, \B, \d, \D match ASCII only
    text_unicode = "Café 123"
    matches_default = re.findall(r"\w+", text_unicode)
    matches_ascii = re.findall(r"\w+", text_unicode, re.ASCII)
    print(f"6. Default: {matches_default}, ASCII only: {matches_ascii}")


# ==========================================
# SECTION 7: COMPILING & PERFORMANCE
# ==========================================

def compilation_and_performance():
    """Pattern compilation and performance considerations."""
    # Compiling patterns for reuse
    pattern = re.compile(r"\b\w{4,}\b")  # Words with 4+ chars
    
    text1 = "The quick brown fox jumps"
    text2 = "Lazy dogs sleep quietly"
    
    matches1 = pattern.findall(text1)
    matches2 = pattern.findall(text2)
    
    print(f"1. Compiled pattern results: {matches1} and {matches2}")
    
    # Accessing compiled pattern attributes
    print(f"2. Pattern details: {pattern.pattern}, Flags: {pattern.flags}")
    
    # Using compiled pattern methods
    match = pattern.search(text1)
    if match:
        print(f"3. Compiled pattern methods: {match.group()}")
    
    # Performance example: pre-compiled vs. one-time use
    import time
    
    # Test string and pattern
    test_text = "a" * 1000 + "b" * 1000
    test_pattern = r"a+b+"
    
    # Time with compilation
    start = time.time()
    compiled = re.compile(test_pattern)
    for _ in range(1000):
        compiled.search(test_text)
    compiled_time = time.time() - start
    
    # Time without compilation
    start = time.time()
    for _ in range(1000):
        re.search(test_pattern, test_text)
    uncompiled_time = time.time() - start
    
    print(f"4. Performance: Compiled: {compiled_time:.4f}s, Uncompiled: {uncompiled_time:.4f}s")
    print(f"   Speedup: {uncompiled_time/compiled_time:.2f}x")


# ==========================================
# SECTION 8: REGEX PATTERNS FOR COMMON TASKS
# ==========================================

def common_patterns():
    """Regex patterns for common text processing tasks."""
    # Email validation
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    
    valid_emails = ["user@example.com", "john.doe@company-name.co.uk"]
    invalid_emails = ["user@.com", "user@com", "@example.com"]
    
    for email in valid_emails + invalid_emails:
        is_valid = email_pattern.fullmatch(email) is not None
        print(f"1. Email '{email}': {'Valid' if is_valid else 'Invalid'}")
    
    # URL matching
    url_pattern = re.compile(r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{2,256}\.[a-z]{2,}\b(?:[-a-zA-Z0-9@:%_+.~#?&/=]*)")
    
    urls = ["https://www.example.com", "http://example.com/path?query=1"]
    for url in urls:
        is_match = url_pattern.match(url) is not None
        print(f"2. URL '{url}': {'Valid' if is_match else 'Invalid'}")
    
    # Date validation (MM/DD/YYYY)
    date_pattern = re.compile(r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})$")
    
    dates = ["12/31/2023", "02/28/2024", "13/01/2023", "01/32/2023"]
    for date in dates:
        is_valid = date_pattern.match(date) is not None
        print(f"3. Date '{date}': {'Valid' if is_valid else 'Invalid'}")
    
    # IP address validation
    ip_pattern = re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
    
    ips = ["192.168.1.1", "255.255.255.255", "192.168.1.256", "192.168.1"]
    for ip in ips:
        is_valid = ip_pattern.match(ip) is not None
        print(f"4. IP '{ip}': {'Valid' if is_valid else 'Invalid'}")
    
    # Password strength check (8+ chars, 1+ uppercase, 1+ lowercase, 1+ digit)
    password_pattern = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$")
    
    passwords = ["Passw0rd", "password", "PASSWORD", "12345678", "Pass"]
    for password in passwords:
        is_strong = password_pattern.match(password) is not None
        print(f"5. Password '{password}': {'Strong' if is_strong else 'Weak'}")


# ==========================================
# SECTION 9: ADVANCED PATTERNS & TECHNIQUES
# ==========================================

def advanced_techniques():
    """Advanced regex patterns and techniques."""
    # Recursive patterns with (?R) - Match nested structures
    # Note: Python's re module doesn't support recursion, this is just an example of the concept
    print("1. Recursive patterns (conceptual example):")
    print("   Pattern: (?R) - Not supported in Python's re module")
    print("   Used for matching nested structures like balanced parentheses")
    print("   Consider using a parsing library for these tasks")
    
    # Atomic grouping (?>...) - No backtracking once matched
    # Note: Python's re module doesn't support atomic grouping directly
    print("2. Atomic grouping (conceptual example):")
    print("   Pattern: (?>...) - Not supported in Python's re module")
    print("   Prevents backtracking for efficiency")
    
    # Conditional patterns (?(condition)yes-pattern|no-pattern)
    # Check if first group captured something
    text = "John: 123-456-7890, Anonymous: unlisted"
    pattern = r"(\w+): (?(1)\d{3}-\d{3}-\d{4}|unlisted)"
    matches = re.findall(pattern, text)
    print(f"3. Conditional pattern results: {matches}")
    
    # Possessive quantifiers - Similar to atomic grouping
    # Note: Python's re module doesn't support possessive quantifiers directly
    print("4. Possessive quantifiers (conceptual example):")
    print("   Patterns like a++, a*+, a?+ - Not supported in Python's re module")
    print("   They prevent backtracking for the quantifier")
    
    # Using negative lookahead for boundary problems
    text = "The great greyhound grazes gracefully"
    # Find all 'gr' words but not those part of longer 'gr' words
    matches = re.findall(r'\bgr\w+?\b', text)
    print(f"5. All 'gr' words: {matches}")
    
    # Word boundaries that aren't space characters
    text = "one,two;three.four"
    matches = re.findall(r'\w+', text)
    print(f"6. Words with non-space boundaries: {matches}")


# ==========================================
# SECTION 10: COMMON PITFALLS & BEST PRACTICES
# ==========================================

def pitfalls_and_best_practices():
    """Common regex pitfalls and best practices."""
    # Pitfall: Greedy vs. non-greedy matching
    text = "<div>Content1</div><div>Content2</div>"
    
    # Greedy matching (default)
    greedy_match = re.search(r"<div>.*</div>", text)
    if greedy_match:
        print(f"1. Greedy match: {greedy_match.group()}")  # Matches everything
    
    # Non-greedy matching (add ?)
    non_greedy_match = re.search(r"<div>.*?</div>", text)
    if non_greedy_match:
        print(f"2. Non-greedy match: {non_greedy_match.group()}")  # Matches first div only
    
    # Pitfall: Character classes vs escape sequences
    text = "The cost is $100."
    
    # Incorrect: \ has special meaning in character classes
    incorrect_match = re.search(r"[$\d]+", text)
    if incorrect_match:
        print(f"3. Incorrect character class usage: {incorrect_match.group()}")
    
    # Correct approach
    correct_match = re.search(r"\$\d+", text)
    if correct_match:
        print(f"4. Correct escape sequence usage: {correct_match.group()}")
    
    # Pitfall: Catastrophic backtracking
    # Example: (a+)+ against "aaaaaaaaaaaaaaaaX" can be very slow
    print("5. Catastrophic backtracking example (conceptual):")
    print("   Pattern: (a+)+ against long string of a's can cause extreme slowdown")
    print("   Solution: Use simpler patterns or limit repetition")
    
    # Best practice: Use raw strings for patterns
    print("6. Raw strings in patterns:")
    print("   Good: r'\\d+' - Escapes are simpler")
    print("   Bad: '\\\\d+' - Double escaping needed")
    
    # Best practice: Compile patterns for reuse
    print("7. Compile patterns for reuse:")
    print("   pattern = re.compile(r'\\d+') - More efficient for multiple uses")
    
    # Best practice: Use more specific patterns
    text = "The ZIP code is 90210"
    
    # Too general: \d+ could match any digits
    general_match = re.search(r"\d+", text)
    if general_match:
        print(f"8. General pattern: {general_match.group()}")
    
    # More specific: constraints on digit count for ZIP codes
    specific_match = re.search(r"\b\d{5}(?:-\d{4})?\b", text)  # ZIP or ZIP+4
    if specific_match:
        print(f"9. Specific pattern: {specific_match.group()}")


# Run all examples
if __name__ == "__main__":
    print("\n===== BASIC MATCHING =====")
    basic_matching()
    
    print("\n===== METACHARACTERS =====")
    metacharacters()
    
    print("\n===== CHARACTER CLASSES =====")
    character_classes()
    
    print("\n===== SPECIAL SEQUENCES =====")
    special_sequences()
    
    print("\n===== SEARCH OPERATIONS =====")
    search_operations()
    
    print("\n===== REPLACE OPERATIONS =====")
    replace_operations()
    
    print("\n===== GROUPS AND CAPTURING =====")
    groups_and_capturing()
    
    print("\n===== LOOKAROUND ASSERTIONS =====")
    lookaround_assertions()
    
    print("\n===== REGEX FLAGS =====")
    regex_flags()
    
    print("\n===== COMPILATION AND PERFORMANCE =====")
    compilation_and_performance()
    
    print("\n===== COMMON PATTERNS =====")
    common_patterns()
    
    print("\n===== ADVANCED TECHNIQUES =====")
    advanced_techniques()
    
    print("\n===== PITFALLS AND BEST PRACTICES =====")
    pitfalls_and_best_practices()


