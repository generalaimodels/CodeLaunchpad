class StringOperations:
    """
    A class to perform various string operations without using external modules.
    """

    def last_index_of(self, string: str, substring: str) -> int:
        """
        Finds the index of the last occurrence of a substring in a string.

        :param string: The string to search in.
        :param substring: The substring to find.
        :return: The index of the last occurrence, or -1 if not found.
        """
        index = -1
        for i in range(len(string) - len(substring), -1, -1):
            match = True
            for j in range(len(substring)):
                if string[i + j] != substring[j]:
                    match = False
                    break
            if match:
                index = i
                break
        return index

    def contains(self, string: str, substring: str) -> bool:
        """
        Checks if the string contains the substring.

        :param string: The string to search in.
        :param substring: The substring to find.
        :return: True if the substring is found, False otherwise.
        """
        return self.last_index_of(string, substring) != -1

    def split_by_delimiter(self, string: str, delimiter: str) -> list:
        """
        Splits the string based on a delimiter.

        :param string: The string to split.
        :param delimiter: The delimiter to split by.
        :return: A list of substrings.
        """
        result = []
        current = ''
        i = 0
        while i < len(string):
            match = True
            for j in range(len(delimiter)):
                if i + j >= len(string) or string[i + j] != delimiter[j]:
                    match = False
                    break
            if match:
                result.append(current)
                current = ''
                i += len(delimiter)
            else:
                current += string[i]
                i += 1
        result.append(current)
        return result

    def escape_special_characters(self, string: str) -> str:
        """
        Escapes special characters in the string by adding backslashes.

        :param string: The string to escape.
        :return: The escaped string.
        """
        special_chars = {'\\': '\\\\', '\n': '\\n', '\t': '\\t', '\r': '\\r', '\"': '\\"', '\'': '\\\''}
        result = ''
        for char in string:
            result += special_chars.get(char, char)
        return result

    def unescape_special_characters(self, string: str) -> str:
        """
        Converts escape sequences to literal characters.

        :param string: The string to unescape.
        :return: The unescaped string.
        """
        result = ''
        i = 0
        while i < len(string):
            if string[i] == '\\' and i + 1 < len(string):
                char = string[i + 1]
                if char == 'n':
                    result += '\n'
                elif char == 't':
                    result += '\t'
                elif char == 'r':
                    result += '\r'
                elif char == '\\':
                    result += '\\'
                elif char == '\"':
                    result += '\"'
                elif char == '\'':
                    result += '\''
                else:
                    result += char
                i += 2
            else:
                result += string[i]
                i += 1
        return result

    # The following methods involve regular expressions.
    # Since we cannot use external modules and writing a full regex engine from scratch is impractical,
    # we will implement limited pattern matching functionality for demonstration purposes.

    def simple_pattern_match(self, string: str, pattern: str) -> bool:
        """
        Checks if the pattern matches the string.
        Supports '*' as any sequence of characters and '?' as any single character.

        :param string: The string to match.
        :param pattern: The pattern to match, supports '*' and '?' wildcards.
        :return: True if the pattern matches the string, False otherwise.
        """
        def match_helper(s_idx, p_idx):
            if p_idx == len(pattern):
                return s_idx == len(string)
            if pattern[p_idx] == '*':
                return (match_helper(s_idx, p_idx + 1) or
                        (s_idx < len(string) and match_helper(s_idx + 1, p_idx)))
            else:
                if s_idx < len(string) and (pattern[p_idx] == string[s_idx] or pattern[p_idx] == '?'):
                    return match_helper(s_idx + 1, p_idx + 1)
                else:
                    return False

        return match_helper(0, 0)

    def replace_first(self, string: str, old: str, new: str) -> str:
        """
        Replaces the first occurrence of a substring with a new substring.

        :param string: The original string.
        :param old: The substring to replace.
        :param new: The new substring.
        :return: The modified string.
        """
        index = self.index_of(string, old)
        if index == -1:
            return string
        return string[:index] + new + string[index + len(old):]

    def replace_all(self, string: str, old: str, new: str) -> str:
        """
        Replaces all occurrences of a substring with a new substring.

        :param string: The original string.
        :param old: The substring to replace.
        :param new: The new substring.
        :return: The modified string.
        """
        result = ''
        i = 0
        while i < len(string):
            match = True
            for j in range(len(old)):
                if i + j >= len(string) or string[i + j] != old[j]:
                    match = False
                    break
            if match:
                result += new
                i += len(old)
            else:
                result += string[i]
                i += 1
        return result

    def index_of(self, string: str, substring: str) -> int:
        """
        Finds the index of the first occurrence of a substring in a string.

        :param string: The string to search in.
        :param substring: The substring to find.
        :return: The index of the first occurrence, or -1 if not found.
        """
        for i in range(len(string) - len(substring) + 1):
            match = True
            for j in range(len(substring)):
                if string[i + j] != substring[j]:
                    match = False
                    break
            if match:
                return i
        return -1

    def greedy_match(self, string: str, pattern: str) -> str:
        """
        Demonstrates greedy matching by matching the longest possible substring.

        :param string: The string to match.
        :param pattern: The pattern to match, supports '*' wildcard.
        :return: The matched substring.
        """
        if pattern == '*':
            return string
        # Since we cannot process full regex, this is a placeholder.
        return ''

    # Additional methods can be added here for other operations.


# Dummy test cases
if __name__ == "__main__":
    operations = StringOperations()

    # Test last_index_of
    print("Testing last_index_of:")
    print(operations.last_index_of("hello world", "o"))  # Expected: 7
    print(operations.last_index_of("hello world", "world"))  # Expected: 6
    print(operations.last_index_of("hello world", "x"))  # Expected: -1

    # Test contains
    print("\nTesting contains:")
    print(operations.contains("hello world", "world"))  # Expected: True
    print(operations.contains("hello world", "x"))  # Expected: False

    # Test split_by_delimiter
    print("\nTesting split_by_delimiter:")
    print(operations.split_by_delimiter("one,two,three", ","))  # Expected: ['one', 'two', 'three']
    print(operations.split_by_delimiter("a b c d", " "))  # Expected: ['a', 'b', 'c', 'd']

    # Test escape_special_characters
    print("\nTesting escape_special_characters:")
    print(operations.escape_special_characters("Hello\nWorld\t!"))  # Expected: 'Hello\\nWorld\\t!'

    # Test unescape_special_characters
    print("\nTesting unescape_special_characters:")
    print(operations.unescape_special_characters("Hello\\nWorld\\t!"))  # Expected: 'Hello\nWorld\t!'

    # Test simple_pattern_match
    print("\nTesting simple_pattern_match:")
    print(operations.simple_pattern_match("hello", "he*o"))  # Expected: True
    print(operations.simple_pattern_match("hello", "he?lo"))  # Expected: True
    print(operations.simple_pattern_match("hello", "he?l?"))  # Expected: False

    # Test replace_first
    print("\nTesting replace_first:")
    print(operations.replace_first("one two three two one", "two", "2"))  # Expected: 'one 2 three two one'

    # Test replace_all
    print("\nTesting replace_all:")
    print(operations.replace_all("one two three two one", "two", "2"))  # Expected: 'one 2 three 2 one'