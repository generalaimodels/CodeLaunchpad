class StringOperations:
    """
    A class to perform various string operations from scratch.
    """

    def __init__(self):
        """
        Initialize the StringOperations class.
        """
        self.string_value = ''
        self.string_pool = {}  # For string interning

    def init_string(self, value):
        """
        String Initialization – Creating a string literal and assigning it.
        """
        self.string_value = value

    def declare_string(self):
        """
        String Declaration – Declaring a string variable.
        """
        self.string_value = ''

    def assign_string(self, value):
        """
        String Assignment – Assigning a string value to a variable.
        """
        self.string_value = value

    def immutable_string_property(self):
        """
        Immutable String Property – Demonstrating that strings are immutable.
        """
        try:
            self.string_value[0] = 'X'
        except TypeError:
            return "Strings are immutable."

    def intern_string(self, value):
        """
        String Interning – Reusing identical string constants to optimize memory.
        """
        if value in self.string_pool:
            return self.string_pool[value]
        else:
            self.string_pool[value] = value
            return value

    def concat_plus(self, str1, str2):
        """
        Concatenation using '+' Operator – Combining two or more strings with the plus operator.
        """
        result = ''
        for char in str1:
            result += char
        for char in str2:
            result += char
        return result

    def concat_join(self, str_list):
        """
        Concatenation using a Join Function – Using a join method to combine string arrays.
        """
        result = ''
        for index, string in enumerate(str_list):
            result += string
        return result

    def string_interpolation(self, template, variables):
        """
        String Interpolation – Embedding variables or expressions within a string.
        """
        result = ''
        i = 0
        while i < len(template):
            if template[i] == '{':
                end_brace = template.find('}', i)
                if end_brace != -1:
                    key = template[i+1:end_brace]
                    result += str(variables.get(key, ''))
                    i = end_brace + 1
                else:
                    result += template[i]
                    i += 1
            else:
                result += template[i]
                i += 1
        return result

    def formatted_string(self, template, *args, **kwargs):
        """
        Formatted Strings – Using format specifiers to create strings.
        """
        result = ''
        arg_index = 0
        i = 0
        while i < len(template):
            if template[i] == '{':
                if template[i+1] == '}':
                    if arg_index < len(args):
                        result += str(args[arg_index])
                        arg_index += 1
                        i += 2
                    else:
                        result += '{}'
                        i += 2
                else:
                    end_brace = template.find('}', i)
                    if end_brace != -1:
                        key = template[i+1:end_brace]
                        result += str(kwargs.get(key, ''))
                        i = end_brace + 1
                    else:
                        result += template[i]
                        i += 1
            else:
                result += template[i]
                i += 1
        return result

    def template_string(self, template, variables):
        """
        Template Strings – Using string templates with placeholders.
        """
        result = ''
        i = 0
        while i < len(template):
            if template[i] == '$':
                if i+1 < len(template) and template[i+1] == '{':
                    end_brace = template.find('}', i)
                    if end_brace != -1:
                        key = template[i+2:end_brace]
                        result += str(variables.get(key, ''))
                        i = end_brace + 1
                    else:
                        result += template[i]
                        i += 1
                else:
                    key = ''
                    j = i + 1
                    while j < len(template) and (template[j].isalnum() or template[j] == '_'):
                        key += template[j]
                        j += 1
                    result += str(variables.get(key, ''))
                    i = j
            else:
                result += template[i]
                i += 1
        return result

    def string_slice(self, start, end):
        """
        String Slicing – Extracting a substring via index ranges.
        """
        result = ''
        for i in range(start, end):
            result += self.string_value[i]
        return result

    def substring(self, start, end):
        """
        Substring Extraction – Extracting part of a string by specifying start and end indices.
        """
        result = ''
        for i in range(start, min(end, len(self.string_value))):
            result += self.string_value[i]
        return result

    def substr(self, start, length):
        """
        Substring Extraction – Extracting a substring using a start index and length.
        """
        result = ''
        for i in range(start, min(start + length, len(self.string_value))):
            result += self.string_value[i]
        return result

    def char_at(self, index):
        """
        Index-Based Character Access – Retrieving a character at a specific index.
        """
        if 0 <= index < len(self.string_value):
            return self.string_value[index]
        else:
            return None

    def iterate_characters(self):
        """
        Character Iteration – Looping through each character in the string.
        """
        chars = []
        for char in self.string_value:
            chars.append(char)
        return chars

    def string_length(self):
        """
        String Length Calculation – Determining the number of characters.
        """
        count = 0
        for _ in self.string_value:
            count += 1
        return count

    def is_empty(self):
        """
        Empty String Check – Testing whether a string has zero length.
        """
        return self.string_length() == 0

    def trim(self):
        """
        Whitespace Trimming (both ends) – Removing leading and trailing spaces.
        """
        start = 0
        end = len(self.string_value) - 1
        while start <= end and self.string_value[start].isspace():
            start += 1
        while end >= start and self.string_value[end].isspace():
            end -= 1
        result = ''
        for i in range(start, end + 1):
            result += self.string_value[i]
        return result

    def ltrim(self):
        """
        Left-Trim Operation – Removing spaces from the start.
        """
        start = 0
        while start < len(self.string_value) and self.string_value[start].isspace():
            start += 1
        result = ''
        for i in range(start, len(self.string_value)):
            result += self.string_value[i]
        return result

    def rtrim(self):
        """
        Right-Trim Operation – Removing spaces from the end.
        """
        end = len(self.string_value) - 1
        while end >= 0 and self.string_value[end].isspace():
            end -= 1
        result = ''
        for i in range(0, end + 1):
            result += self.string_value[i]
        return result

    def remove_whitespace(self):
        """
        Complete Whitespace Removal – Removing all whitespace characters.
        """
        result = ''
        for char in self.string_value:
            if not char.isspace():
                result += char
        return result

    def to_uppercase(self):
        """
        Case Conversion to Uppercase – Converting all characters to uppercase.
        """
        result = ''
        for char in self.string_value:
            ascii_code = ord(char)
            if 97 <= ascii_code <= 122:
                result += chr(ascii_code - 32)
            else:
                result += char
        return result

    def to_lowercase(self):
        """
        Case Conversion to Lowercase – Converting all characters to lowercase.
        """
        result = ''
        for char in self.string_value:
            ascii_code = ord(char)
            if 65 <= ascii_code <= 90:
                result += chr(ascii_code + 32)
            else:
                result += char
        return result

    def capitalize(self):
        """
        Capitalization – Changing the first character to uppercase.
        """
        if not self.string_value:
            return ''
        result = ''
        first_char = self.string_value[0]
        ascii_code = ord(first_char)
        if 97 <= ascii_code <= 122:
            result += chr(ascii_code - 32)
        else:
            result += first_char
        for i in range(1, len(self.string_value)):
            result += self.string_value[i]
        return result

    def title_case(self):
        """
        Title Case Conversion – Converting the first letter of each word to uppercase.
        """
        result = ''
        new_word = True
        for char in self.string_value:
            if char.isspace():
                new_word = True
                result += char
            else:
                if new_word:
                    ascii_code = ord(char)
                    if 97 <= ascii_code <= 122:
                        result += chr(ascii_code - 32)
                    else:
                        result += char
                    new_word = False
                else:
                    result += char
        return result

    def swap_case(self):
        """
        Case Inversion – Swapping uppercase to lowercase and vice versa.
        """
        result = ''
        for char in self.string_value:
            ascii_code = ord(char)
            if 65 <= ascii_code <= 90:
                result += chr(ascii_code + 32)
            elif 97 <= ascii_code <= 122:
                result += chr(ascii_code - 32)
            else:
                result += char
        return result

    def starts_with(self, prefix):
        """
        Checking if String Starts with a Substring – Using a starts-with function.
        """
        if len(prefix) > len(self.string_value):
            return False
        for i in range(len(prefix)):
            if self.string_value[i] != prefix[i]:
                return False
        return True

    def ends_with(self, suffix):
        """
        Checking if String Ends with a Substring – Using an ends-with function.
        """
        if len(suffix) > len(self.string_value):
            return False
        start_index = len(self.string_value) - len(suffix)
        for i in range(len(suffix)):
            if self.string_value[start_index + i] != suffix[i]:
                return False
        return True

    def index_of(self, substring):
        """
        IndexOf Operation – Finding the index of the first occurrence of a substring.
        """
        for i in range(len(self.string_value) - len(substring) + 1):
            match = True
            for j in range(len(substring)):
                if self.string_value[i + j] != substring[j]:
                    match = False
                    break
            if match:
                return i
        return -1

    def last_index_of(self, substring):
        """
        LastIndexOf Operation – Finding the index of the last occurrence.
        """
        for i in range(len(self.string_value) - len(substring), -1, -1):
            match = True
            for j in range(len(substring)):
                if self.string_value[i + j] != substring[j]:
                    match = False
                    break
            if match:
                return i
        return -1


# Dummy test cases for each method developed
if __name__ == "__main__":
    operations = StringOperations()

    # String Initialization
    operations.init_string("Hello World")
    print("Initialized String:", operations.string_value)

    # String Declaration
    operations.declare_string()
    print("Declared String:", operations.string_value)

    # String Assignment
    operations.assign_string("Python")
    print("Assigned String:", operations.string_value)

    # Immutable String Property
    print("Immutable String Test:", operations.immutable_string_property())

    # String Interning
    str_interned1 = operations.intern_string("intern")
    str_interned2 = operations.intern_string("intern")
    print("String Interning:", str_interned1 is str_interned2)

    # Concatenation using '+' Operator
    concat_plus_result = operations.concat_plus("Hello ", "World")
    print("Concatenation with '+':", concat_plus_result)

    # Concatenation using Join Function
    concat_join_result = operations.concat_join(["Hello", " ", "World"])
    print("Concatenation with join:", concat_join_result)

    # String Interpolation
    interpolation_result = operations.string_interpolation("Hello, {name}!", {"name": "Alice"})
    print("String Interpolation:", interpolation_result)

    # Formatted Strings
    formatted_string_result = operations.formatted_string("Coordinates: ({}, {})", 10, 20)
    print("Formatted String:", formatted_string_result)

    # Template Strings
    template_string_result = operations.template_string("Hello, ${name}!", {"name": "Bob"})
    print("Template String:", template_string_result)

    # String Slicing
    operations.assign_string("Hello World")
    slice_result = operations.string_slice(0, 5)
    print("String Slicing:", slice_result)

    # Substring Extraction (substring method)
    substring_result = operations.substring(6, 11)
    print("Substring Extraction:", substring_result)

    # Substring Extraction (substr method)
    substr_result = operations.substr(6, 5)
    print("Substr Extraction:", substr_result)

    # Index-Based Character Access
    char_at_result = operations.char_at(1)
    print("Character at Index 1:", char_at_result)

    # Character Iteration
    chars = operations.iterate_characters()
    print("Character Iteration:", chars)

    # String Length Calculation
    length = operations.string_length()
    print("String Length:", length)

    # Empty String Check
    empty_check = operations.is_empty()
    print("Is String Empty:", empty_check)

    # Whitespace Trimming (both ends)
    operations.assign_string("   Trim me   ")
    trimmed = operations.trim()
    print("Trimmed String:", "'" + trimmed + "'")

    # Left-Trim Operation
    ltrimmed = operations.ltrim()
    print("Left-Trimmed String:", "'" + ltrimmed + "'")

    # Right-Trim Operation
    rtrimmed = operations.rtrim()
    print("Right-Trimmed String:", "'" + rtrimmed + "'")

    # Complete Whitespace Removal
    no_whitespace = operations.remove_whitespace()
    print("Whitespace Removed:", "'" + no_whitespace + "'")

    # Case Conversion to Uppercase
    operations.assign_string("Hello World")
    uppercase_result = operations.to_uppercase()
    print("Uppercase Conversion:", uppercase_result)

    # Case Conversion to Lowercase
    lowercase_result = operations.to_lowercase()
    print("Lowercase Conversion:", lowercase_result)

    # Capitalization
    capitalized_result = operations.capitalize()
    print("Capitalization:", capitalized_result)

    # Title Case Conversion
    title_case_result = operations.title_case()
    print("Title Case Conversion:", title_case_result)

    # Case Inversion
    swapped_case_result = operations.swap_case()
    print("Case Inversion:", swapped_case_result)

    # Checking if String Starts with a Substring
    starts_with_result = operations.starts_with("Hello")
    print("Starts with 'Hello':", starts_with_result)

    # Checking if String Ends with a Substring
    ends_with_result = operations.ends_with("World")
    print("Ends with 'World':", ends_with_result)

    # IndexOf Operation
    index_of_result = operations.index_of("World")
    print("Index of 'World':", index_of_result)

    # LastIndexOf Operation
    last_index_of_result = operations.last_index_of("l")
    print("Last Index of 'l':", last_index_of_result)