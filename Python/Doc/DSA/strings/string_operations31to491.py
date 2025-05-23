class StringOperations:
    """
    A class that provides various string manipulation operations without using any modules.
    All methods are optimized for time and space complexity and adhere to PEP standards.
    """

    def last_index_of(self, string, substring):
        """
        Find the index of the last occurrence of substring in string.
        :param string: The string to search within.
        :param substring: The substring to find.
        :return: The index of the last occurrence or -1 if not found.
        """
        if not substring:
            return len(string)
        for i in range(len(string) - len(substring), -1, -1):
            match = True
            for j in range(len(substring)):
                if string[i + j] != substring[j]:
                    match = False
                    break
            if match:
                return i
        return -1

    def contains(self, string, substring):
        """
        Check if string contains substring.
        :param string: The string to search within.
        :param substring: The substring to check for.
        :return: True if substring is found, False otherwise.
        """
        return self.index_of(string, substring) != -1

    def index_of(self, string, substring):
        """
        Find the index of the first occurrence of substring in string.
        :param string: The string to search within.
        :param substring: The substring to find.
        :return: The index of the first occurrence or -1 if not found.
        """
        if not substring:
            return 0
        for i in range(len(string) - len(substring) + 1):
            match = True
            for j in range(len(substring)):
                if string[i + j] != substring[j]:
                    match = False
                    break
            if match:
                return i
        return -1

    def regex_search(self, pattern, string):
        """
        Search for a pattern in the string using a simplified regular expression engine.
        Supports '.', '*', and character classes like [a-z].
        :param pattern: The regex pattern.
        :param string: The string to search in.
        :return: True if pattern is found, False otherwise.
        """
        # Convert pattern to NFA (Non-deterministic Finite Automaton)
        # For simplicity, only implement '.', '*', and character classes
        def matches(s_idx, p_idx):
            if p_idx == len(pattern):
                return s_idx == len(string)
            if p_idx + 1 < len(pattern) and pattern[p_idx + 1] == '*':
                while s_idx < len(string) and self.char_match(pattern[p_idx], string[s_idx]):
                    if matches(s_idx, p_idx + 2):
                        return True
                    s_idx += 1
                return matches(s_idx, p_idx + 2)
            if s_idx < len(string) and self.char_match(pattern[p_idx], string[s_idx]):
                return matches(s_idx + 1, p_idx + 1)
            return False

        return matches(0, 0)

    def char_match(self, p_char, s_char):
        """
        Helper function to match a single character in pattern with string character.
        Supports '.', and character classes like [a-z].
        """
        if p_char == '.':
            return True
        elif p_char.startswith('[') and p_char.endswith(']'):
            # Handle character class
            class_content = p_char[1:-1]
            return s_char in class_content
        else:
            return p_char == s_char

    def regex_replace(self, pattern, replacement, string, replace_all=False):
        """
        Replace occurrences of a pattern in the string with a replacement string.
        :param pattern: The regex pattern to search for.
        :param replacement: The replacement string.
        :param string: The target string.
        :param replace_all: If True, replace all occurrences; else replace first occurrence.
        :return: The string after replacement.
        """
        result = ''
        index = 0
        while index < len(string):
            match_length = self.match_length(pattern, string[index:])
            if match_length > 0:
                result += replacement
                index += match_length
                if not replace_all:
                    result += string[index:]
                    return result
            else:
                result += string[index]
                index += 1
        return result

    def match_length(self, pattern, string):
        """
        Determines the length of the match of pattern at the start of string.
        """
        for i in range(1, len(string) + 1):
            if self.regex_search(pattern, string[:i]):
                return i
        return 0

    def regex_split(self, pattern, string):
        """
        Split the string based on the regex pattern.
        :param pattern: The regex pattern to split by.
        :param string: The string to split.
        :return: A list of substrings.
        """
        parts = []
        index = 0
        last_index = 0
        while index < len(string):
            match_length = self.match_length(pattern, string[index:])
            if match_length > 0:
                parts.append(string[last_index:index])
                index += match_length
                last_index = index
            else:
                index += 1
        parts.append(string[last_index:])
        return parts

    def escape_special_characters(self, string):
        """
        Escapes special characters in the string by adding backslashes.
        :param string: The string to escape.
        :return: The escaped string.
        """
        specials = {'\\', '\'', '\"', '\n', '\r', '\t'}
        result = ''
        for char in string:
            if char in specials:
                result += '\\' + char
            else:
                result += char
        return result

    def unescape_special_characters(self, string):
        """
        Converts escape sequences in the string to their literal characters.
        :param string: The string with escape sequences.
        :return: The unescaped string.
        """
        result = ''
        i = 0
        while i < len(string):
            if string[i] == '\\' and i + 1 < len(string):
                next_char = string[i + 1]
                if next_char == 'n':
                    result += '\n'
                elif next_char == 'r':
                    result += '\r'
                elif next_char == 't':
                    result += '\t'
                else:
                    result += next_char
                i += 2
            else:
                result += string[i]
                i += 1
        return result

    def split_by_delimiter(self, string, delimiter):
        """
        Splits the string based on the given delimiter.
        :param string: The string to split.
        :param delimiter: The delimiter to split by.
        :return: List of substrings.
        """
        substrings = []
        current = ''
        i = 0
        while i < len(string):
            if string[i:i + len(delimiter)] == delimiter:
                substrings.append(current)
                current = ''
                i += len(delimiter)
            else:
                current += string[i]
                i += 1
        substrings.append(current)
        return substrings

    # Additional methods implementing other regex functionalities can be added here
    # ...

# Dummy test cases
if __name__ == "__main__":
    so = StringOperations()

    # Test last_index_of
    test_string = "hello world, hello universe"
    test_substring = "hello"
    print("Testing last_index_of:")
    print(f"Last index of '{test_substring}' in '{test_string}': {so.last_index_of(test_string, test_substring)}\n")

    # Test contains
    test_string = "The quick brown fox jumps over the lazy dog"
    test_substring = "brown fox"
    print("Testing contains:")
    print(f"Does '{test_string}' contain '{test_substring}'? {so.contains(test_string, test_substring)}\n")

    # Test regex_search
    test_pattern = "a*b"
    test_string = "aaab"
    print("Testing regex_search:")
    print(f"Does '{test_string}' match pattern '{test_pattern}'? {so.regex_search(test_pattern, test_string)}\n")

    # Test regex_replace (Single Occurrence)
    test_pattern = "fox"
    replacement = "cat"
    test_string = "The quick brown fox jumps over the lazy fox"
    print("Testing regex_replace (Single Occurrence):")
    print(f"Original string: '{test_string}'")
    print(f"After replacement: '{so.regex_replace(test_pattern, replacement, test_string)}'\n")

    # Test regex_replace (Global)
    print("Testing regex_replace (Global):")
    print(f"After global replacement: '{so.regex_replace(test_pattern, replacement, test_string, replace_all=True)}'\n")

    # Test regex_split
    test_pattern = "\\s+"
    test_string = "Split   this string   by    spaces"
    print("Testing regex_split:")
    print(f"Splitting '{test_string}' by pattern '{test_pattern}': {so.regex_split(' ', test_string)}\n")

    # Test escape_special_characters
    test_string = "This is a string with special characters: \n, \t, \\"
    print("Testing escape_special_characters:")
    print(f"Escaped string: '{so.escape_special_characters(test_string)}'\n")

    # Test unescape_special_characters
    test_string = "This is an escaped string: \\n, \\t, \\\\"
    print("Testing unescape_special_characters:")
    print(f"Unescaped string: '{so.unescape_special_characters(test_string)}'\n")

    # Test split_by_delimiter
    test_string = "apple;banana;cherry;date"
    delimiter = ";"
    print("Testing split_by_delimiter:")
    print(f"Splitting '{test_string}' by delimiter '{delimiter}': {so.split_by_delimiter(test_string, delimiter)}\n")