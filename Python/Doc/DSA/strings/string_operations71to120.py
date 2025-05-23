# StringOperations.py

class StringOperations:
    """
    A class encapsulating various string operations.
    """

    def find_all_occurrences(self, string: str, substring: str) -> list:
        """
        Locates every instance of a substring in a string.
        Returns a list of starting indices where the substring is found.
        """
        indices = []
        index = string.find(substring)
        while index != -1:
            indices.append(index)
            index = string.find(substring, index + 1)
        return indices

    def replace_based_on_mapping(self, string: str, mapping: dict) -> str:
        """
        Replaces characters in a string based on a dictionary mapping.
        """
        result = ''
        for char in string:
            result += mapping.get(char, char)
        return result

    def transliterate(self, string: str, source: str, target: str) -> str:
        """
        Converts characters from one alphabet to another based on provided source and target alphabets.
        The source and target must be strings of the same length.
        """
        mapping = {s: t for s, t in zip(source, target)}
        result = ''
        for char in string:
            result += mapping.get(char, char)
        return result

    def filter_characters_by_condition(self, string: str, condition) -> str:
        """
        Removes characters from the string based on a condition function.
        The condition function should return True for characters to keep.
        """
        return ''.join([char for char in string if condition(char)])

    def remove_punctuation(self, string: str) -> str:
        """
        Removes all punctuation marks from the string.
        """
        punctuation_marks = '!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        return ''.join([char for char in string if char not in punctuation_marks])

    def remove_digits(self, string: str) -> str:
        """
        Eliminates all numeric characters from the string.
        """
        return ''.join([char for char in string if not char.isdigit()])

    def remove_non_alphanumeric(self, string: str) -> str:
        """
        Keeps only letters and digits in the string.
        """
        return ''.join([char for char in string if char.isalnum()])

    def truncate_string(self, string: str, length: int) -> str:
        """
        Cuts off the string at a specified length.
        """
        return string[:length]

    def append_ellipsis(self, string: str, length: int) -> str:
        """
        Truncates the string and adds "..." if it exceeds the specified length.
        """
        return (string[:length] + '...') if len(string) > length else string

    def extract_substring_by_delimiters(self, string: str, start_delim: str, end_delim: str) -> str:
        """
        Gets part of the string between two delimiters.
        """
        start_index = string.find(start_delim)
        end_index = string.find(end_delim, start_index + len(start_delim))
        if start_index != -1 and end_index != -1:
            return string[start_index + len(start_delim):end_index]
        return ''

    def parse_key_value_pairs(self, string: str, item_delim: str, key_value_delim: str) -> dict:
        """
        Extracts keys and values from a formatted string.
        """
        result = {}
        items = string.split(item_delim)
        for item in items:
            if key_value_delim in item:
                key, value = item.split(key_value_delim, 1)
                result[key.strip()] = value.strip()
        return result

    def csv_line_parsing(self, line: str) -> list:
        """
        Converts a CSV-formatted line into a list of values.
        """
        values = []
        current = ''
        in_quotes = False
        for char in line:
            if char == '"' and not in_quotes:
                in_quotes = True
            elif char == '"' and in_quotes:
                in_quotes = False
            elif char == ',' and not in_quotes:
                values.append(current)
                current = ''
            else:
                current += char
        values.append(current)
        return values

    def csv_generation(self, values: list) -> str:
        """
        Converts a list of values into a CSV-formatted line.
        """
        result = []
        for value in values:
            value_str = str(value)
            if ',' in value_str or '"' in value_str:
                value_str = '"' + value_str.replace('"', '""') + '"'
            result.append(value_str)
        return ','.join(result)

    def html_escape(self, string: str) -> str:
        """
        Converts characters to HTML-safe sequences.
        """
        escape_mapping = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
        }
        return ''.join([escape_mapping.get(char, char) for char in string])

    def html_unescape(self, string: str) -> str:
        """
        Converts HTML entities back to characters.
        """
        unescape_mapping = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
        }
        import re
        pattern = re.compile('|'.join(re.escape(key) for key in unescape_mapping.keys()))
        return pattern.sub(lambda x: unescape_mapping[x.group()], string)

    def url_encode(self, string: str) -> str:
        """
        Converts a string to a URL-safe format.
        """
        safe_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~'
        result = ''
        for char in string:
            if char in safe_chars:
                result += char
            else:
                result += '%' + hex(ord(char))[2:].upper()
        return result

    def url_decode(self, string: str) -> str:
        """
        Reverts URL encoding to original characters.
        """
        result = ''
        i = 0
        length = len(string)
        while i < length:
            if string[i] == '%':
                hex_value = string[i+1:i+3]
                result += chr(int(hex_value, 16))
                i += 3
            else:
                result += string[i]
                i += 1
        return result

    def base64_encode(self, string: str) -> str:
        """
        Encodes a string into Base64.
        """
        # Base64 characters
        base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        binary_string = ''
        for char in string:
            binary_char = bin(ord(char))[2:].zfill(8)
            binary_string += binary_char
        padding = '=' * ((6 - len(binary_string) % 6) % 6)
        binary_string += '0' * ((6 - len(binary_string) % 6) % 6)
        encoded = ''
        for i in range(0, len(binary_string), 6):
            sextet = binary_string[i:i+6]
            index = int(sextet, 2)
            encoded += base64_chars[index]
        encoded += padding
        return encoded

    def base64_decode(self, encoded: str) -> str:
        """
        Decodes a Base64 string.
        """
        base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        padding = encoded.count('=')
        encoded = encoded.rstrip('=')
        binary_string = ''
        for char in encoded:
            index = base64_chars.index(char)
            binary_char = bin(index)[2:].zfill(6)
            binary_string += binary_char
        # Remove padding bits
        if padding:
            binary_string = binary_string[:-(padding * 2)]
        decoded = ''
        for i in range(0, len(binary_string), 8):
            byte = binary_string[i:i+8]
            decoded += chr(int(byte, 2))
        return decoded

    def unicode_normalization_nfc(self, string: str) -> str:
        """
        Normalizes a Unicode string using NFC (Normalization Form C).
        """
        # Since we cannot use modules, we'll implement a simple NFC normalization
        # For demonstration purposes, this will replace composed characters
        # with their single code point equivalents where possible.
        # In practice, proper normalization requires a comprehensive mapping.
        normalization_map = {
            'á': 'á',
            'é': 'é',
            'í': 'í',
            'ó': 'ó',
            'ú': 'ú',
            'ñ': 'ñ',
        }
        result = ''
        for char in string:
            result += normalization_map.get(char, char)
        return result

    def unicode_code_point_extraction(self, string: str) -> list:
        """
        Gets the Unicode code points of characters in the string.
        """
        return [ord(char) for char in string]

    def iterating_over_unicode_code_points(self, string: str):
        """
        Loops over the Unicode code points, yielding code point values.
        """
        for char in string:
            yield ord(char)

    def hashing_md5(self, string: str) -> str:
        """
        Generates an MD5 hash of the string.
        """
        # Since we cannot use modules, we'll implement a simple MD5-like hash
        # This is NOT a real MD5 hash and should not be used for cryptographic purposes.
        # This is for demonstration only.
        hash_value = 0
        for char in string:
            hash_value = (hash_value * 31 + ord(char)) % (2**128)
        return hex(hash_value)[2:].zfill(32)

    def hashing_sha1(self, string: str) -> str:
        """
        Generates a SHA1 hash of the string.
        """
        # Since we cannot use modules, we'll implement a simple SHA1-like hash
        # This is NOT a real SHA1 hash and should not be used for cryptographic purposes.
        # This is for demonstration only.
        hash_value = 0
        for char in string:
            hash_value = (hash_value * 131 + ord(char)) % (2**160)
        return hex(hash_value)[2:].zfill(40)

    def hashing_sha256(self, string: str) -> str:
        """
        Generates a SHA256 hash of the string.
        """
        # Since we cannot use modules, we'll implement a simple SHA256-like hash
        # This is NOT a real SHA256 hash and should not be used for cryptographic purposes.
        # This is for demonstration only.
        hash_value = 0
        for char in string:
            hash_value = (hash_value * 257 + ord(char)) % (2**256)
        return hex(hash_value)[2:].zfill(64)

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculates the Levenshtein distance between two strings.
        """
        len_s1, len_s2 = len(s1), len(s2)
        dp = [[0]*(len_s2+1) for _ in range(len_s1+1)]
        for i in range(len_s1+1):
            dp[i][0] = i
        for j in range(len_s2+1):
            dp[0][j] = j
        for i in range(1, len_s1+1):
            for j in range(1, len_s2+1):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
        return dp[len_s1][len_s2]

    def hamming_distance(self, s1: str, s2: str) -> int:
        """
        Calculates the Hamming distance between two equal-length strings.
        """
        if len(s1) != len(s2):
            raise ValueError("Strings must be of equal length.")
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def longest_common_subsequence(self, s1: str, s2: str) -> str:
        """
        Finds the longest common subsequence between two strings.
        """
        len_s1, len_s2 = len(s1), len(s2)
        dp = [['']*(len_s2+1) for _ in range(len_s1+1)]
        for i in range(len_s1):
            for j in range(len_s2):
                if s1[i] == s2[j]:
                    dp[i+1][j+1] = dp[i][j] + s1[i]
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j], key=len)
        return dp[len_s1][len_s2]

    def longest_common_substring(self, s1: str, s2: str) -> str:
        """
        Finds the longest common substring between two strings.
        """
        len_s1, len_s2 = len(s1), len(s2)
        longest = 0
        x_longest = 0
        dp = [[0]*(len_s2+1) for _ in range(len_s1+1)]
        for x in range(1, len_s1+1):
            for y in range(1, len_s2+1):
                if s1[x-1] == s2[y-1]:
                    dp[x][y] = dp[x-1][y-1] + 1
                    if dp[x][y] > longest:
                        longest = dp[x][y]
                        x_longest = x
                else:
                    dp[x][y] = 0
        return s1[x_longest-longest: x_longest]

    def jaro_winkler_distance(self, s1: str, s2: str) -> float:
        """
        Calculates the Jaro-Winkler distance between two strings.
        """
        # Helper function to calculate Jaro similarity
        def jaro_similarity(s1, s2):
            if s1 == s2:
                return 1.0

            len_s1 = len(s1)
            len_s2 = len(s2)

            max_dist = (max(len_s1, len_s2) // 2) - 1

            match = 0
            hash_s1 = [0] * len_s1
            hash_s2 = [0] * len_s2

            for i in range(len_s1):
                for j in range(max(0, i - max_dist), min(len_s2, i + max_dist + 1)):
                    if s1[i] == s2[j] and hash_s2[j] == 0:
                        hash_s1[i] = 1
                        hash_s2[j] = 1
                        match += 1
                        break

            if match == 0:
                return 0.0

            t = 0
            point = 0

            for i in range(len_s1):
                if hash_s1[i]:
                    while hash_s2[point] == 0:
                        point +=1
                    if s1[i] != s2[point]:
                        t += 1
                    point += 1
            t = t//2

            return (match / len_s1 + match / len_s2 + (match - t) / match)/ 3.0

        jaro_dist = jaro_similarity(s1, s2)
        prefix = 0

        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        prefix = min(4, prefix)
        jaro_winkler_sim = jaro_dist + 0.1 * prefix * (1 - jaro_dist)
        return jaro_winkler_sim

# Dummy test cases
def main():
    ops = StringOperations()

    # Test find_all_occurrences
    print("Find All Occurrences:")
    print(ops.find_all_occurrences("abracadabra", "abra"))  # [0, 7]

    # Test replace_based_on_mapping
    print("\nReplace Based on Mapping:")
    print(ops.replace_based_on_mapping("hello world", {'h': 'H', 'w': 'W'}))  # "Hello World"

    # Test transliteration
    print("\nTransliteration:")
    print(ops.transliterate("héllö", "éö", "eo"))  # "hello"

    # Test filter_characters_by_condition
    print("\nFilter Characters by Condition:")
    print(ops.filter_characters_by_condition("abc123", lambda c: c.isalpha()))  # "abc"

    # Test remove_punctuation
    print("\nRemove Punctuation:")
    print(ops.remove_punctuation("Hello, world!"))  # "Hello world"

    # Test remove_digits
    print("\nRemove Digits:")
    print(ops.remove_digits("a1b2c3"))  # "abc"

    # Test remove_non_alphanumeric
    print("\nRemove Non-Alphanumeric:")
    print(ops.remove_non_alphanumeric("hello@world.com"))  # "helloworldcom"

    # Test truncate_string
    print("\nTruncate String:")
    print(ops.truncate_string("Hello, world!", 5))  # "Hello"

    # Test append_ellipsis
    print("\nAppend Ellipsis:")
    print(ops.append_ellipsis("Hello, world!", 5))  # "Hello..."

    # Test extract_substring_by_delimiters
    print("\nExtract Substring by Delimiters:")
    print(ops.extract_substring_by_delimiters("Start[ExtractThis]End", "[", "]"))  # "ExtractThis"

    # Test parse_key_value_pairs
    print("\nParse Key-Value Pairs:")
    print(ops.parse_key_value_pairs("key1=value1;key2=value2", ";", "="))  # {'key1': 'value1', 'key2': 'value2'}

    # Test CSV Line Parsing
    print("\nCSV Line Parsing:")
    print(ops.csv_line_parsing('value1,"value, with, commas",value3'))  # ['value1', 'value, with, commas', 'value3']

    # Test CSV Generation
    print("\nCSV Generation:")
    print(ops.csv_generation(['value1', 'value, with, commas', 'value3']))  # 'value1,"value, with, commas",value3'

    # Test HTML Escaping
    print("\nHTML Escaping:")
    print(ops.html_escape('<div class="content">Hello & Welcome</div>'))  # '&lt;div class=&quot;content&quot;&gt;Hello &amp; Welcome&lt;/div&gt;'

    # Test HTML Unescaping
    print("\nHTML Unescaping:")
    print(ops.html_unescape('&lt;div&gt;Test&amp;Demo&lt;/div&gt;'))  # '<div>Test&Demo</div>'

    # Test URL Encoding
    print("\nURL Encoding:")
    print(ops.url_encode('hello world'))  # 'hello%20world'

    # Test URL Decoding
    print("\nURL Decoding:")
    print(ops.url_decode('hello%20world'))  # 'hello world'

    # Test Base64 Encoding
    print("\nBase64 Encoding:")
    print(ops.base64_encode('hello'))  # 'aGVsbG8='

    # Test Base64 Decoding
    print("\nBase64 Decoding:")
    print(ops.base64_decode('aGVsbG8='))  # 'hello'

    # Test Unicode Normalization NFC
    print("\nUnicode Normalization NFC:")
    print(ops.unicode_normalization_nfc('café'))  # 'café'

    # Test Unicode Code Point Extraction
    print("\nUnicode Code Point Extraction:")
    print(ops.unicode_code_point_extraction('ABC'))  # [65, 66, 67]

    # Test Iterating Over Unicode Code Points
    print("\nIterating Over Unicode Code Points:")
    for code_point in ops.iterating_over_unicode_code_points('ABC'):
        print(code_point, end=' ')  # '65 66 67'
    print()

    # Test Hashing MD5
    print("\nHashing MD5:")
    print(ops.hashing_md5('hello'))  # Simulated hash value

    # Test Hashing SHA1
    print("\nHashing SHA1:")
    print(ops.hashing_sha1('hello'))  # Simulated hash value

    # Test Hashing SHA256
    print("\nHashing SHA256:")
    print(ops.hashing_sha256('hello'))  # Simulated hash value

    # Test Levenshtein Distance
    print("\nLevenshtein Distance:")
    print(ops.levenshtein_distance('kitten', 'sitting'))  # 3

    # Test Hamming Distance
    print("\nHamming Distance:")
    print(ops.hamming_distance('karolin', 'kathrin'))  # 3

    # Test Longest Common Subsequence
    print("\nLongest Common Subsequence:")
    print(ops.longest_common_subsequence('ABCBDAB', 'BDCAB'))  # 'BDAB'

    # Test Longest Common Substring
    print("\nLongest Common Substring:")
    print(ops.longest_common_substring('abcdxyz', 'xyzabcd'))  # 'abcd'

    # Test Jaro-Winkler Distance
    print("\nJaro-Winkler Distance:")
    print(ops.jaro_winkler_distance('MARTHA', 'MARHTA'))  # Similarity score

if __name__ == "__main__":
    main()