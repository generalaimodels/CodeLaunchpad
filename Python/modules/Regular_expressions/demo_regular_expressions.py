
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Guide to Python's re Module for Regular Expressions

This module provides an extensive exploration of the re module,
which implements regular expression matching operations.
"""

import re

# =============================================================================
# SECTION 1: INTRODUCTION TO REGULAR EXPRESSIONS
# =============================================================================

"""
Regular expressions (regex) are powerful sequences of characters that define
search patterns. The re module in Python provides operations for working with
regular expressions, enabling complex string searching, matching, and manipulation.

Key concepts:
- Patterns: Define what you're searching for
- Functions: Operations that use patterns (search, match, findall, etc.)
- Special characters: Have specific meanings in regex patterns
- Flags: Modify how patterns are interpreted
"""

# Basic example - finding a simple pattern in text
text = "Python is a programming language. python is versatile."
pattern = r"Python"  # r prefix creates a raw string - recommended for regex
# breakpoint()

# The search() function looks for the first match of the pattern
search_result = re.search(pattern, text)
print(f"Basic search result: {search_result}")  # Shows match object with position
print(f"Matched text: {search_result.group()}")  # The actual matched text
print(f"Start position: {search_result.start()}")  # Start index of the match
print(f"End position: {search_result.end()}")  # End index of the match

# =============================================================================
# SECTION 2: CORE RE MODULE FUNCTIONS
# =============================================================================

"""
The re module provides several key functions for working with regular expressions:

1. re.search(pattern, string, flags=0)
   - Scans through the string looking for the first match of the pattern
   - Returns a Match object if found, None otherwise

2. re.match(pattern, string, flags=0)
   - Checks if the pattern matches at the beginning of the string
   - Returns a Match object if found, None otherwise

3. re.fullmatch(pattern, string, flags=0)
   - Checks if the pattern matches the entire string
   - Returns a Match object if found, None otherwise

4. re.findall(pattern, string, flags=0)
   - Returns all non-overlapping matches as a list of strings
   - If the pattern contains groups, returns a list of tuples

5. re.finditer(pattern, string, flags=0)
   - Returns an iterator yielding Match objects for all matches

6. re.split(pattern, string, maxsplit=0, flags=0)
   - Splits the string by occurrences of the pattern
   - Returns a list of substrings

7. re.sub(pattern, repl, string, count=0, flags=0)
   - Replaces occurrences of the pattern with repl
   - Returns the modified string

8. re.subn(pattern, repl, string, count=0, flags=0)
   - Like sub(), but returns a tuple (new_string, number_of_subs_made)

9. re.escape(pattern)
   - Escapes special characters in a string for use in a regex pattern

10. re.compile(pattern, flags=0)
    - Compiles a regex pattern for improved performance when used multiple times
"""

example_text = "Python was created in 1991. Python 2 was released in 2000. Python 3 came in 2008."

# Example 1: re.search() - finds first occurrence
search_result = re.search(r"Python \d", example_text)
print(f"\nre.search example: {search_result.group()}")  # Output: Python 2

# Example 2: re.match() - only matches at the beginning of the string
match_result = re.match(r"Python", example_text)
print(f"re.match at beginning: {match_result.group() if match_result else 'No match'}")  # Python

match_result_middle = re.match(r"Python \d", example_text)
print(f"re.match in middle: {match_result_middle.group() if match_result_middle else 'No match'}")  # No match

# Example 3: re.fullmatch() - the pattern must match the entire string
full_match = re.fullmatch(r"Python", "Python")
print(f"re.fullmatch exact: {full_match.group() if full_match else 'No match'}")  # Python

full_match_partial = re.fullmatch(r"Python", "Python 3")
print(f"re.fullmatch partial: {full_match_partial.group() if full_match_partial else 'No match'}")  # No match

# Example 4: re.findall() - finds all occurrences
findall_result = re.findall(r"Python \d", example_text)
print(f"re.findall: {findall_result}")  # ['Python 2', 'Python 3']

# Example 5: re.finditer() - iterator of match objects
print("re.finditer results:")
for match in re.finditer(r"Python \d", example_text):
    print(f"  Found '{match.group()}' at position {match.start()}-{match.end()}")

# Example 6: re.split() - split string by pattern
split_result = re.split(r"\. ", example_text)
print(f"re.split: {split_result}")  # ['Python was created in 1991', 'Python 2 was released in 2000', 'Python 3 came in 2008.']

# Example 7: re.sub() - substitute pattern with replacement
sub_result = re.sub(r"Python (\d)", r"Python version \1", example_text)
print(f"re.sub: {sub_result}")  # Python was created in 1991. Python version 2 was released in 2000. Python version 3 came in 2008.

# Example 8: re.subn() - substitute with count
subn_result = re.subn(r"Python", "Ruby", example_text, count=2)
print(f"re.subn: {subn_result}")  # (New string, number of replacements)

# Example 9: re.escape() - escape special regex characters
special_chars = ".*+?^$()[]{}|\\"
escaped = re.escape(special_chars)
print(f"re.escape: '{special_chars}' becomes '{escaped}'")

# Example 10: re.compile() - compile pattern for reuse
python_pattern = re.compile(r"Python \d")
compile_results = python_pattern.findall(example_text)
print(f"re.compile with findall: {compile_results}")  # ['Python 2', 'Python 3']

# =============================================================================
# SECTION 3: REGEX PATTERN SYNTAX - BASIC CHARACTERS
# =============================================================================

"""
Regular expression patterns use special characters to define search criteria:

1. Basic characters:
   - Most characters match themselves (a-z, A-Z, 0-9)
   - Some characters are special and need to be escaped with \ to match literally

2. Most common special characters and their meanings:
   - . (dot): Matches any character except newline
   - \: Escape character for special chars or introducing special sequences
   - ^ (caret): Matches start of string
   - $: Matches end of string
   - |: Alternation (OR operator)
"""

basic_text = "Python costs $0 and matches the pattern."

# Literal character matching
print("\nBASIC CHARACTER MATCHING:")
print(f"Finding 'Python': {re.search(r'Python', basic_text).group()}")  # Python

# The dot (.) matches any character except newline
print(f"Using dot (.) to match any char: {re.search(r'P.thon', basic_text).group()}")  # Python
print(f"Multiple dots: {re.search(r'c..ts', basic_text).group()}")  # costs

# Escape special characters with backslash
# print(f"Escaping $ sign: {re.search(r'\$0', basic_text).group()}")  # $0

# ^ matches the beginning of a string
start_match = re.search(r'^Python', basic_text)
not_start_match = re.search(r'^costs', basic_text)
print(f"^ at start: {start_match.group() if start_match else 'No match'}")  # Python
print(f"^ not at start: {not_start_match.group() if not_start_match else 'No match'}")  # No match

# $ matches the end of a string
end_match = re.search(r'pattern\.$', basic_text)
not_end_match = re.search(r'Python$', basic_text)
print(f"$ at end: {end_match.group() if end_match else 'No match'}")  # pattern.
print(f"$ not at end: {not_end_match.group() if not_end_match else 'No match'}")  # No match

# | for alternation (OR)
or_match = re.search(r'Python|Java', basic_text)
print(f"Alternation: {or_match.group()}")  # Python

multiple_or = re.findall(r'Python|costs|\$0|pattern', basic_text)
print(f"Multiple alternations: {multiple_or}")  # ['Python', 'costs', '$0', 'pattern']

# =============================================================================
# SECTION 4: CHARACTER CLASSES
# =============================================================================

"""
Character classes define sets of characters to match:

1. [...] - Custom character class:
   - [abc]: Matches 'a', 'b', or 'c'
   - [a-z]: Matches any lowercase letter
   - [0-9]: Matches any digit
   - [^abc]: Matches any character EXCEPT a, b, or c

2. Predefined character classes:
   - \d: Matches any digit [0-9]
   - \D: Matches any non-digit [^0-9]
   - \w: Matches any word character [a-zA-Z0-9_]
   - \W: Matches any non-word character
   - \s: Matches any whitespace character (space, tab, newline, etc.)
   - \S: Matches any non-whitespace character
"""

char_class_text = "Python3 has 2 major versions. Test_123! Contains-hyphen."

print("\nCHARACTER CLASSES:")

# Custom character classes with []
print(f"Vowels [aeiou]: {re.findall(r'[aeiou]', char_class_text)}")  # List of all vowels
print(f"Consonants [b-df-hj-np-tv-z]: {re.findall(r'[b-df-hj-np-tv-z]', char_class_text[:10])}")  # Consonants in "Python3 ha"

# Range in character class
print(f"Letters a-m: {re.findall(r'[a-m]', char_class_text[:10])}")  # Letters a through m
print(f"Digits in text: {re.findall(r'[0-9]', char_class_text)}")  # All digits

# Negated character class with ^
print(f"Not digits or letters [^a-zA-Z0-9]: {re.findall(r'[^a-zA-Z0-9]', char_class_text)}")  # Non-alphanumeric chars

# Predefined character classes
# print(f"\\d (digits): {re.findall(r'\d', char_class_text)}")  # All digits
# print(f"\\D (non-digits): {re.findall(r'\D', char_class_text[:10])}")  # All non-digits in "Python3 ha"
# print(f"\\w (word chars): {re.findall(r'\w', 'abc_123.!?')}")  # Word characters
# print(f"\\W (non-word): {re.findall(r'\W', 'abc_123.!?')}")  # Non-word characters
# print(f"\\s (whitespace): {re.findall(r'\s', char_class_text)}")  # All whitespace
# print(f"\\S (non-whitespace): {re.findall(r'\S', 'a b c')}")  # Non-whitespace chars

# Combining character classes
print(f"Digits or vowels [0-9aeiou]: {re.findall(r'[0-9aeiou]', char_class_text[:15])}")  # Digits or vowels
print(f"Word followed by digit: {re.findall(r'[a-zA-Z]+[0-9]', char_class_text)}")  # Word followed by digit

# =============================================================================
# SECTION 5: QUANTIFIERS
# =============================================================================

"""
Quantifiers specify how many times a character or group should match:

1. Basic quantifiers:
   - *: 0 or more times (greedy)
   - +: 1 or more times (greedy)
   - ?: 0 or 1 time (greedy)
   - {n}: Exactly n times
   - {n,}: n or more times
   - {n,m}: Between n and m times (inclusive)

2. Greedy vs. Non-greedy (lazy) matching:
   - Greedy: Matches as much as possible
   - Non-greedy: Matches as little as possible, add ? after quantifier
     Examples: *?, +?, ??, {n,}?, {n,m}?
"""

quant_text = "aaa 123 abc12345 xy z test_999 test-123-456"

print("\nQUANTIFIERS:")

# * (0 or more)
print(f"'a*' (zero or more a's): {re.findall(r'a*', 'aaa bcd')}")  # ['aaa', '', '', '', '']
print(f"'xa*y' (x followed by 0+ a's then y): {re.findall(r'xa*y', 'xy xay xaaay xbcd')}")  # ['xy', 'xay', 'xaaay']

# + (1 or more)
print(f"'a+' (one or more a's): {re.findall(r'a+', 'aaa bcd')}")  # ['aaa']
print(f"'xa+y' (x followed by 1+ a's then y): {re.findall(r'xa+y', 'xy xay xaaay xbcd')}")  # ['xay', 'xaaay']

# ? (0 or 1)
print(f"'colou?r' (optional u): {re.findall(r'colou?r', 'color colour')}")  # ['color', 'colour']
print(f"'a?' (zero or one a): {re.findall(r'xa?y', 'xy xay xaaay')}")  # ['xy', 'xay']

# {n} (exactly n)
print(f"'a{{3}}' (exactly 3 a's): {re.findall(r'a{3}', 'a aa aaa aaaa')}")  # ['aaa', 'aaa']

# {n,} (n or more)
print(f"'a{{2,}}' (2 or more a's): {re.findall(r'a{2,}', 'a aa aaa aaaa')}")  # ['aa', 'aaa', 'aaaa']

# {n,m} (between n and m)
print(f"'a{{2,3}}' (2 to 3 a's): {re.findall(r'a{2,3}', 'a aa aaa aaaa')}")  # ['aa', 'aaa', 'aaa']

# Greedy vs. non-greedy
html = "<div>Content 1</div><div>Content 2</div>"

# Greedy: matches as much as possible
greedy = re.search(r'<div>.*</div>', html)
print(f"Greedy match: {greedy.group()}")  # <div>Content 1</div><div>Content 2</div>

# Non-greedy: matches as little as possible
non_greedy = re.search(r'<div>.*?</div>', html)
print(f"Non-greedy match: {non_greedy.group()}")  # <div>Content 1</div>

# Combined examples
# print(f"Words with digits: {re.findall(r'\w+\d+', quant_text)}")  # ['abc12345', 'test_999']
# print(f"Digit sequences: {re.findall(r'\d+', quant_text)}")  # ['123', '12345', '999', '123', '456']
# print(f"3 to 5 digits: {re.findall(r'\d{3,5}', quant_text)}")  # ['123', '12345', '999', '123', '456']

# =============================================================================
# SECTION 6: GROUPING AND CAPTURING
# =============================================================================

"""
Grouping allows you to:
1. Apply quantifiers to entire patterns
2. Capture matched text for later use
3. Create back-references

Key concepts:
- (...): Define a capture group
- (?:...): Define a non-capture group (doesn't store the match)
- \n: Backreference to the nth captured group
- (?P<name>...): Named capture group
- (?P=name): Reference to a named group
"""

group_text = "John Smith (john.smith@example.com). Jane Doe (jane.doe@company.co.uk)."

print("\nGROUPING AND CAPTURING:")

# Basic groups with ()
basic_match = re.search(r'(\w+) (\w+)', group_text)
print(f"Full match: {basic_match.group(0)}")  # John Smith
print(f"Group 1: {basic_match.group(1)}")  # John
print(f"Group 2: {basic_match.group(2)}")  # Smith
print(f"All groups: {basic_match.groups()}")  # ('John', 'Smith')

# Nested groups
nested = re.search(r'((\w+) (\w+))', group_text)
print(f"Outer group: {nested.group(1)}")  # John Smith
print(f"Inner group 1: {nested.group(2)}")  # John
print(f"Inner group 2: {nested.group(3)}")  # Smith

# Grouping for applying quantifiers
emails = re.findall(r'(\w+\.\w+@\w+\.\w+(\.\w+)?)', group_text)
print(f"Emails with domain groups: {emails}")  # [('john.smith@example.com', ''), ('jane.doe@company.co.uk', '.co.uk')]

# Non-capturing group (?:...)
non_capturing = re.findall(r'(?:\w+\.\w+@\w+)(\.\w+(?:\.\w+)?)', group_text)
print(f"Only domains (non-capturing): {non_capturing}")  # ['.com', '.co.uk']

# Backreferences with \n
repeated_words = re.findall(r'(\b\w+\b) \1', "The the quick brown fox jumps over the lazy dog")
print(f"Repeated words: {repeated_words}")  # ['The']

html_tags = re.findall(r'<(\w+)>.*?</\1>', "<h1>Title</h1> <p>Paragraph</p>")
print(f"HTML tags: {html_tags}")  # ['h1', 'p']

# Named groups with (?P<name>...)
named_match = re.search(r'(?P<first>\w+) (?P<last>\w+)', group_text)
print(f"Named group 'first': {named_match.group('first')}")  # John
print(f"Named group 'last': {named_match.group('last')}")  # Smith

# Named backreference with (?P=name)
pythonic = re.search(r'(?P<word>\w+)_\d+ (?P=word)', "test_123 test other_456 other")
print(f"Named backreference: {pythonic.group() if pythonic else 'No match'}")  # "other_456 other"

# Practical example: parsing emails with groups
email_pattern = r'(\w+)\.(\w+)@(\w+)\.(\w+(?:\.\w+)?)'
email_matches = re.findall(email_pattern, group_text)
print(f"Parsed emails: {email_matches}")
# [('john', 'smith', 'example', 'com'), ('jane', 'doe', 'company', 'co.uk')]

for match in email_matches:
    print(f"  First: {match[0]}, Last: {match[1]}, Domain: {match[2]}.{match[3]}")

# =============================================================================
# SECTION 7: LOOKAHEAD AND LOOKBEHIND ASSERTIONS
# =============================================================================

"""
Lookahead and lookbehind assertions match positions, not characters:

1. Lookahead assertions:
   - Positive (?=...): Matches if pattern ahead matches
   - Negative (?!...): Matches if pattern ahead doesn't match

2. Lookbehind assertions:
   - Positive (?<=...): Matches if pattern behind matches
   - Negative (?<!...): Matches if pattern behind doesn't match

Note: These assertions don't consume characters in the string.
"""

assert_text = "price: $1000, discount: $100, total: $900"

print("\nLOOKAHEAD AND LOOKBEHIND ASSERTIONS:")

# Positive lookahead (?=...)
# Find all digits followed by 'price'
price_lookahead = re.findall(r'\d+(?=.*?price)', assert_text[::-1])
# Note: We're reversing the string since our price comes before digits in the original text
print(f"Digits followed by 'price' (reversed string): {[x[::-1] for x in price_lookahead]}")  # ['1000']

# More useful lookahead - find $ followed by a price
dollars = re.findall(r'\$(?=\d+)', assert_text)
print(f"$ symbols before digits: {dollars}")  # ['$', '$', '$']

# Negative lookahead (?!...)
# Find all digits NOT followed by '0'
not_tens = re.findall(r'\d(?!\d*0$)', assert_text)
print(f"Digits not part of a number ending in 0: {not_tens}")  # ['1', '0', '0', '9', '0']

# Positive lookbehind (?<=...)
# Find prices (digits after $)
prices = re.findall(r'(?<=\$)\d+', assert_text)
print(f"Prices: {prices}")  # ['1000', '100', '900']

# Negative lookbehind (?<!...)
# Find numbers not preceded by $
not_prices = re.findall(r'(?<!\$)\d+', "item1 has price: $50 and item2 is 30 percent off")
print(f"Numbers not preceded by $: {not_prices}")  # ['1', '2', '30']

# Combined lookahead and lookbehind
# Find prices between $100 and $999
mid_range = re.findall(r'(?<=\$)[1-9]\d{2}(?!\d)', assert_text)
print(f"Prices $100-$999: {mid_range}")  # ['100', '900']

# Using with word boundaries
words_not_after_total = re.findall(r'(?<!total: )\b\w+:', assert_text)
print(f"Labels not after 'total': {words_not_after_total}")  # ['price:', 'discount:']

# Practical example: extract values with specific format
csv_data = "name,age,score\nJohn,25,95\nJane,22,87"
csv_values = re.findall(r'(?<=,)\d+(?=,|\n|$)', csv_data)
print(f"CSV numeric values: {csv_values}")  # ['25', '95', '22', '87']

# =============================================================================
# SECTION 8: FLAGS
# =============================================================================

"""
Regex flags modify how patterns are interpreted:

1. re.IGNORECASE or re.I: Case-insensitive matching
2. re.MULTILINE or re.M: ^ and $ match start/end of each line
3. re.DOTALL or re.S: Dot (.) matches any character including newline
4. re.VERBOSE or re.X: Allows comments and whitespace in patterns
5. re.ASCII or re.A: Makes \w, \W, \b, \B, \d, \D, \s, \S match ASCII only
6. re.LOCALE or re.L: Make \w, \W, \b, \B, \d, \D, \s, \S depend on locale
7. re.DEBUG: Display debug information

Flags can be specified:
- As an argument to functions: re.search(pattern, string, flags=re.I)
- Inside the pattern using (?flags): (?i)pattern
"""

flag_text = """Python is great.
python is versatile.
PYTHON is powerful."""

print("\nFLAGS:")

# re.IGNORECASE (re.I) - Case-insensitive matching
case_sensitive = re.findall(r'python', flag_text)
case_insensitive = re.findall(r'python', flag_text, re.IGNORECASE)
print(f"Case sensitive: {case_sensitive}")  # ['python']
print(f"Case insensitive: {case_insensitive}")  # ['Python', 'python', 'PYTHON']

# Inline flag for case-insensitive
inline_case = re.findall(r'(?i)python', flag_text)
print(f"Inline case-insensitive: {inline_case}")  # ['Python', 'python', 'PYTHON']

# re.MULTILINE (re.M) - ^ and $ match start/end of lines
starts_with_p = re.findall(r'^p\w+', flag_text, re.MULTILINE | re.IGNORECASE)
print(f"Start of line with 'p': {starts_with_p}")  # ['Python', 'python', 'PYTHON']

# re.DOTALL (re.S) - . matches newline too
without_dotall = re.findall(r'great.*?versatile', flag_text)
with_dotall = re.findall(r'great.*?versatile', flag_text, re.DOTALL)
print(f"Without DOTALL: {without_dotall}")  # []
print(f"With DOTALL: {with_dotall}")  # ['great.\npython is versatile']

# re.VERBOSE (re.X) - Allow whitespace and comments in patterns
phone_pattern = re.compile(r'''
    \((\d{3})\)  # Area code
    \s*          # Optional whitespace
    (\d{3})      # First 3 digits
    -            # Separator
    (\d{4})      # Last 4 digits
''', re.VERBOSE)

phone_match = phone_pattern.search("Call (123) 456-7890 today")
print(f"VERBOSE pattern match: {phone_match.groups() if phone_match else 'No match'}")  # ('123', '456', '7890')

# re.ASCII (re.A) - Limit \w, \d, etc. to ASCII range
text_with_unicode = "café 123"
# print(f"ASCII-only word chars: {re.findall(r'\w+', text_with_unicode, re.ASCII)}")  # ['caf', '123']
# print(f"Unicode word chars: {re.findall(r'\w+', text_with_unicode)}")  # ['café', '123']

# Multiple flags
multi_flags = re.findall(r'is.*?\.', flag_text, re.DOTALL | re.IGNORECASE)
print(f"Multiple flags: {multi_flags}")  # ['is great.\npython is versatile.\nPYTHON is powerful.']

# =============================================================================
# SECTION 9: COMMON USE CASES
# =============================================================================

"""
This section demonstrates practical regex patterns for common scenarios:
1. Email validation
2. URL parsing
3. Phone number formatting
4. Date parsing
5. HTML tag extraction
6. CSV parsing
7. Password validation
8. IP address matching
"""

print("\nCOMMON USE CASES:")

# Email validation
def validate_email(email):
    """Validate email address format."""
    # Basic pattern for demonstration
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

emails = ["user@example.com", "invalid@.com", "name@domain.co.uk", "no_at_sign.com"]
for email in emails:
    print(f"Email '{email}' is {'valid' if validate_email(email) else 'invalid'}")

# URL parsing
def parse_url(url):
    """Extract components from a URL."""
    pattern = r'^(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?:/([^?#]*))?(?:\?([^#]*))?(?:#(.*))?$'
    match = re.match(pattern, url)
    if match:
        domain, path, query, fragment = match.groups()
        return {
            "domain": domain,
            "path": path if path else "",
            "query": query if query else "",
            "fragment": fragment if fragment else ""
        }
    return None

url = "https://www.example.com/path/to/page?name=value&x=y#section"
url_parts = parse_url(url)
print(f"URL parts: {url_parts}")

# Phone number formatting
def format_phone(phone):
    """Format various phone number inputs to (XXX) XXX-XXXX."""
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return re.sub(r'(\d{3})(\d{3})(\d{4})', r'(\1) \2-\3', digits)
    elif len(digits) == 11 and digits[0] == '1':
        return re.sub(r'1(\d{3})(\d{3})(\d{4})', r'(\1) \2-\3', digits)
    return "Invalid phone number"

phones = ["1234567890", "(123) 456-7890", "123-456-7890", "1-123-456-7890", "12345"]
for phone in phones:
    print(f"Formatted '{phone}': {format_phone(phone)}")

# Date parsing
def parse_date(date_str):
    """Parse dates in various formats to YYYY-MM-DD."""
    patterns = [
        # MM/DD/YYYY
        r'(\d{1,2})/(\d{1,2})/(\d{4})',
        # DD-MM-YYYY
        r'(\d{1,2})-(\d{1,2})-(\d{4})',
        # Month DD, YYYY
        r'([A-Za-z]+) (\d{1,2}), (\d{4})'
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.match(pattern, date_str)
        if match:
            if i == 0:  # MM/DD/YYYY
                mm, dd, yyyy = match.groups()
                return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}"
            elif i == 1:  # DD-MM-YYYY
                dd, mm, yyyy = match.groups()
                return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}"
            elif i == 2:  # Month DD, YYYY
                month_names = ["January", "February", "March", "April", "May", "June",
                              "July", "August", "September", "October", "November", "December"]
                month, dd, yyyy = match.groups()
                for j, name in enumerate(month_names, 1):
                    if name.lower().startswith(month.lower()):
                        return f"{yyyy}-{str(j).zfill(2)}-{dd.zfill(2)}"
    return "Invalid date format"

dates = ["12/31/2023", "31-12-2023", "December 31, 2023", "2023/12/31"]
for date in dates:
    print(f"Parsed date '{date}': {parse_date(date)}")

# HTML tag extraction
def extract_html_tags(html):
    """Extract HTML tags and their content."""
    pattern = r'<([a-zA-Z0-9]+)(?:\s+[^>]*)?>(.*?)</\1>'
    return re.findall(pattern, html, re.DOTALL)

html_content = "<div class='container'><h1>Title</h1><p>This is a paragraph.</p></div>"
tags = extract_html_tags(html_content)
print(f"HTML tags: {tags}")

# CSV parsing
def parse_csv_line(line):
    """Parse a CSV line, handling quoted fields with commas."""
    pattern = r',(?=(?:[^"]*"[^"]*")*[^"]*$)'
    return re.split(pattern, line)

csv_line = 'John,Doe,"123 Main St, Apt 4",555-1234'
parsed_csv = parse_csv_line(csv_line)
print(f"CSV fields: {parsed_csv}")

# Password validation
def validate_password(password):
    """
    Validate password strength:
    - At least 8 characters
    - Contains uppercase letter
    - Contains lowercase letter
    - Contains a digit
    - Contains a special character
    """
    checks = [
        (r'.{8,}', "At least 8 characters"),
        (r'[A-Z]', "Has uppercase letter"),
        (r'[a-z]', "Has lowercase letter"),
        (r'\d', "Has digit"),
        (r'[^A-Za-z0-9]', "Has special character")
    ]
    
    results = []
    for pattern, description in checks:
        if re.search(pattern, password):
            results.append(f"✓ {description}")
        else:
            results.append(f"✗ {description}")
    
    return results

passwords = ["pass", "Password1", "p@ssw0rd", "P@ssw0rd!"]
for password in passwords:
    print(f"Password '{password}' validation:")
    for result in validate_password(password):
        print(f"  {result}")

# IP address validation
def validate_ip(ip):
    """Validate IPv4 address."""
    pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return bool(re.match(pattern, ip))

ips = ["192.168.1.1", "256.0.0.1", "10.0.0", "172.16.0.1"]
for ip in ips:
    print(f"IP '{ip}' is {'valid' if validate_ip(ip) else 'invalid'}")

# =============================================================================
# SECTION 10: PERFORMANCE OPTIMIZATION AND BEST PRACTICES
# =============================================================================

"""
Best practices and optimization techniques for regular expressions:

1. Compile patterns for reuse
2. Use non-capturing groups when possible
3. Avoid unnecessary backtracking
4. Be specific with character classes
5. Use appropriate quantifiers
6. Use lookahead/lookbehind judiciously
7. Test patterns on edge cases
8. Be cautious with nested quantifiers (can lead to catastrophic backtracking)
"""

import time

print("\nPERFORMANCE OPTIMIZATION AND BEST PRACTICES:")

# Example 1: Compiling patterns for reuse
def demo_compile_benefit():
    text = "Python programming language" * 1000
    
    # Approach 1: Compile once, reuse
    start_time = time.time()
    pattern = re.compile(r'Python')
    for _ in range(1000):
        pattern.search(text)
    compiled_time = time.time() - start_time
    
    # Approach 2: Recompile each time
    start_time = time.time()
    for _ in range(1000):
        re.search(r'Python', text)
    uncompiled_time = time.time() - start_time
    
    print(f"Compiled pattern time: {compiled_time:.6f}s")
    print(f"Uncompiled pattern time: {uncompiled_time:.6f}s")
    # print(f"Speedup: {uncompiled_time/compiled_time:.2f}x faster with compilation")

demo_compile_benefit()

# Example 2: Non-capturing groups vs. capturing groups
def demo_noncapturing_groups():
    text = "Python programming language" * 1000
    
    # With capturing groups
    start_time = time.time()
    pattern_capturing = re.compile(r'(Python|Java) (programming|coding)')
    for _ in range(1000):
        pattern_capturing.search(text)
    capturing_time = time.time() - start_time
    
    # With non-capturing groups
    start_time = time.time()
    pattern_noncapturing = re.compile(r'(?:Python|Java) (?:programming|coding)')
    for _ in range(1000):
        pattern_noncapturing.search(text)
    noncapturing_time = time.time() - start_time
    
    print(f"Capturing groups time: {capturing_time:.6f}s")
    print(f"Non-capturing groups time: {noncapturing_time:.6f}s")
    print(f"Improvement: {capturing_time/noncapturing_time:.2f}x faster with non-capturing groups")

demo_noncapturing_groups()

# Example 3: Avoiding catastrophic backtracking
def demo_backtracking_issue():
    # Nested quantifiers can cause exponential backtracking
    bad_pattern = re.compile(r'(a+)+b')  # Problematic pattern
    good_pattern = re.compile(r'a+b')    # Better pattern
    
    test_string = "a" * 20  # String with only 'a's, no 'b'
    
    # Good pattern
    start_time = time.time()
    good_pattern.search(test_string)
    good_time = time.time() - start_time
    
    # Bad pattern - may be extremely slow due to backtracking
    # Limiting to just a shorter string to avoid hanging
    short_test = "a" * 10
    start_time = time.time()
    bad_pattern.search(short_test)
    bad_time = time.time() - start_time
    
    print(f"Good pattern time: {good_time:.6f}s")
    print(f"Potentially problematic pattern time (shorter input): {bad_time:.6f}s")
    print("Note: Bad patterns with nested quantifiers can cause 'catastrophic backtracking'")
    print("      and should be avoided for performance-critical applications.")

demo_backtracking_issue()

# Best practices summary
print("\nBEST PRACTICES SUMMARY:")
best_practices = [
    "1. Compile regex patterns that are used multiple times",
    "2. Use non-capturing groups (?:...) when you don't need the matched content",
    "3. Be specific with character classes instead of using wildcard .* patterns",
    "4. Avoid nested quantifiers like (a+)+ which can cause catastrophic backtracking",
    "5. Use possessive quantifiers (a++, a*+) or atomic groups (?>...) when available",
    "6. Prefer non-greedy quantifiers (*?, +?) when appropriate",
    "7. Use anchors (^, $, \\b) to fix positions and improve performance",
    "8. Test your regex on edge cases and with performance benchmarks",
    "9. Break complex patterns into simpler ones when possible",
    "10. Consider alternatives to regex for extremely complex text processing tasks"
]

for practice in best_practices:
    print(f"  {practice}")

# =============================================================================
# SECTION 11: HANDLING EXCEPTIONS
# =============================================================================

"""
When working with regex, several types of exceptions can occur:

1. re.error: Raised when an invalid regex pattern is provided
2. TypeError: Raised when incorrect types are passed to regex functions
3. IndexError: When accessing non-existent groups
4. AttributeError: When calling methods on None (from a failed match)
"""

print("\nHANDLING EXCEPTIONS:")

# 1. Handling re.error (invalid pattern)
def safe_regex(pattern, text):
    """Safely execute a regex pattern."""
    try:
        return re.search(pattern, text)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

valid_result = safe_regex(r'[a-z]+', "hello")
invalid_result = safe_regex(r'[a-z', "hello")  # Missing closing bracket
print(f"Valid pattern result: {valid_result.group() if isinstance(valid_result, re.Match) else valid_result}")
print(f"Invalid pattern result: {invalid_result}")

# 2. Handling TypeError
try:
    re.search(123, "hello")  # Pattern must be a string
except TypeError as e:
    print(f"TypeError: {e}")

# 3. Handling IndexError with groups
text = "Hello world"
match = re.search(r'(Hello) (world)', text)
try:
    print(f"Group 3: {match.group(3)}")  # No group 3
except IndexError as e:
    print(f"IndexError: {e}")

# 4. Handling AttributeError with failed matches
no_match = re.search(r'Python', "Java")
try:
    print(f"Match: {no_match.group()}")  # no_match is None
except AttributeError as e:
    print(f"AttributeError: {e}")

# Proper way to check for matches
pattern = r'Python'
text = "Java"
match = re.search(pattern, text)
if match:
    print(f"Found: {match.group()}")
else:
    print(f"Pattern '{pattern}' not found in '{text}'")

# Safe group access
def safe_group(match, group_num):
    """Safely access a match group."""
    if not match:
        return None
    try:
        return match.group(group_num)
    except IndexError:
        return None

match = re.search(r'(\d+)', "Age: 30")
print(f"Safe group 1: {safe_group(match, 1)}")  # 30
print(f"Safe group 2: {safe_group(match, 2)}")  # None

# =============================================================================
# SECTION 12: ADDITIONAL REGEX FEATURES
# =============================================================================

"""
Some additional regex features that are useful:

1. Atomic grouping (?>...): Once matched, the group is "locked" and won't be backtracked
2. POSIX character classes: [:alpha:], [:digit:], etc.
3. Conditional patterns: (?(id/name)true-pattern|false-pattern)
4. Unicode properties: \p{} (in Python 3.8+)
5. Special escapes: \A, \Z, \b, \B
"""

print("\nADDITIONAL REGEX FEATURES:")

# Word boundaries with \b and \B
boundary_text = "word boundary and unbound"
# print(f"Words with 'bound' at boundary: {re.findall(r'\bbound\w*', boundary_text)}")  # ['boundary']
# print(f"'bound' not at word boundary: {re.findall(r'\Bbound\w*', boundary_text)}")  # ['unbound']

# \A and \Z - match start and end of string (not affected by MULTILINE)
string_anchors = re.search(r'\APython.*end\Z', "Python at the start and end", re.MULTILINE)
print(f"\\A and \\Z anchors: {string_anchors.group() if string_anchors else 'No match'}")

# Match vs. Search difference
match_demo = "Second line contains Python"
print(f"re.match for 'Python': {re.match(r'Python', match_demo) is not None}")  # False - not at beginning
print(f"re.search for 'Python': {re.search(r'Python', match_demo) is not None}")  # True - found in string

# Unicode support in Python 3
unicode_text = "你好, Python! こんにちは 😊"
# print(f"Unicode characters: {re.findall(r'\w+', unicode_text)}")  # ['你好', 'Python', 'こんにちは']

# Counting match occurrences
text_to_count = "Python is popular. Python is powerful. Python is easy to learn."
python_count = len(re.findall(r'\bPython\b', text_to_count))
print(f"'Python' appears {python_count} times")

# Using sub with function
def add_units(match):
    """Add appropriate unit to a number based on size."""
    num = int(match.group(1))
    if num == 1:
        return f"{num} item"
    return f"{num} items"

inventory = "We have 1 apple, 3 oranges, and 5 bananas."
with_units = re.sub(r'(\d+)', add_units, inventory)
print(f"With units: {with_units}")

# Matching balanced parentheses (limited recursive capability)
def find_balanced_parens(text):
    """Find simple balanced parentheses - non-recursive approach for demo."""
    level = 0
    start = None
    results = []
    
    for i, char in enumerate(text):
        if char == '(':
            if level == 0:
                start = i
            level += 1
        elif char == ')':
            level -= 1
            if level == 0 and start is not None:
                results.append(text[start:i+1])
    
    return results

paren_text = "This (is (a) test) with (multiple) parentheses."
balanced = find_balanced_parens(paren_text)
print(f"Balanced parentheses: {balanced}")

# Handling multiline patterns with re.VERBOSE
complex_pattern = re.compile(r'''
    # Match a date in YYYY-MM-DD format
    (\d{4})  # Year
    -        # Separator
    (0[1-9]|1[0-2])  # Month (01-12)
    -        # Separator
    (0[1-9]|[12]\d|3[01])  # Day (01-31)
''', re.VERBOSE)

date_match = complex_pattern.search("Today is 2023-05-15")
if date_match:
    year, month, day = date_match.groups()
    print(f"Date components: year={year}, month={month}, day={day}")
