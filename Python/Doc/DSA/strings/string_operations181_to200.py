#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
String Operations Module

This module provides a collection of classes and methods to perform various
string operations without the use of external libraries. Each operation is
implemented with attention to detail, optimized for both time and space,
and follows PEP 8 standards.

Author: OpenAI ChatGPT (Assistant)
"""

class StringParser:
    """
    Class for parsing and converting strings and numbers.
    """

    def parse_int(self, s):
        """
        Parses a string and converts it to an integer.

        Args:
            s (str): The string representation of an integer.

        Returns:
            int: The integer value of the string.

        Raises:
            ValueError: If the string is not a valid integer.
        """
        s = s.strip()
        if not s:
            raise ValueError("Empty string cannot be converted to integer.")
        negative = False
        index = 0
        if s[0] == '-':
            negative = True
            index += 1
        elif s[0] == '+':
            index += 1
        result = 0
        while index < len(s):
            if '0' <= s[index] <= '9':
                result = result * 10 + (ord(s[index]) - ord('0'))
            else:
                raise ValueError("Invalid character found: '{}'".format(s[index]))
            index += 1
        return -result if negative else result

    def parse_float(self, s):
        """
        Parses a string and converts it to a floating-point number.

        Args:
            s (str): The string representation of a float.

        Returns:
            float: The floating-point value of the string.

        Raises:
            ValueError: If the string is not a valid float.
        """
        s = s.strip()
        if not s:
            raise ValueError("Empty string cannot be converted to float.")
        negative = False
        index = 0
        if s[0] == '-':
            negative = True
            index += 1
        elif s[0] == '+':
            index += 1

        integer_part = 0
        fractional_part = 0
        fractional_divisor = 1
        has_fraction = False

        while index < len(s):
            if '0' <= s[index] <= '9':
                if not has_fraction:
                    integer_part = integer_part * 10 + (ord(s[index]) - ord('0'))
                else:
                    fractional_part = fractional_part * 10 + (ord(s[index]) - ord('0'))
                    fractional_divisor *= 10
            elif s[index] == '.':
                if has_fraction:
                    raise ValueError("Invalid float format.")
                has_fraction = True
            else:
                raise ValueError("Invalid character found: '{}'".format(s[index]))
            index += 1

        result = integer_part + fractional_part / fractional_divisor
        return -result if negative else result

    def int_to_string(self, num):
        """
        Converts an integer to its string representation.

        Args:
            num (int): The integer to convert.

        Returns:
            str: The string representation of the integer.
        """
        if num == 0:
            return '0'
        negative = num < 0
        num = abs(num)
        digits = []
        while num > 0:
            digits.append(chr(ord('0') + num % 10))
            num //= 10
        if negative:
            digits.append('-')
        return ''.join(reversed(digits))

    def float_to_string(self, num, precision=6):
        """
        Converts a float to its string representation.

        Args:
            num (float): The float to convert.
            precision (int): Number of decimal places.

        Returns:
            str: The string representation of the float.
        """
        negative = num < 0
        num = abs(num)
        integer_part = int(num)
        fractional_part = num - integer_part
        integer_str = self.int_to_string(integer_part)

        fractional_digits = []
        for _ in range(precision):
            fractional_part *= 10
            digit = int(fractional_part)
            fractional_digits.append(chr(ord('0') + digit))
            fractional_part -= digit

        result = integer_str + '.' + ''.join(fractional_digits)
        return '-' + result if negative else result

class ArithmeticEvaluator:
    """
    Class for evaluating arithmetic expressions represented as strings.
    """

    def evaluate(self, expression):
        """
        Evaluates a simple arithmetic expression.

        Supports +, -, *, / operators and integer numbers.

        Args:
            expression (str): The arithmetic expression.

        Returns:
            float: The result of the expression.

        Raises:
            ValueError: If the expression is invalid.
        """
        tokens = self.tokenize(expression)
        postfix = self.infix_to_postfix(tokens)
        result = self.evaluate_postfix(postfix)
        return result

    def tokenize(self, s):
        """
        Converts a string expression into tokens.

        Args:
            s (str): The expression string.

        Returns:
            list: A list of tokens.
        """
        tokens = []
        i = 0
        while i < len(s):
            if s[i].isspace():
                i += 1
                continue
            elif s[i] in '+-*/()':
                tokens.append(s[i])
                i += 1
            elif s[i].isdigit() or s[i] == '.':
                num = []
                dot_count = 0
                while i < len(s) and (s[i].isdigit() or s[i] == '.'):
                    if s[i] == '.':
                        dot_count += 1
                        if dot_count > 1:
                            raise ValueError("Invalid number format.")
                    num.append(s[i])
                    i += 1
                tokens.append(''.join(num))
            else:
                raise ValueError("Invalid character found: '{}'".format(s[i]))
        return tokens

    def infix_to_postfix(self, tokens):
        """
        Converts infix expression tokens to postfix notation.

        Args:
            tokens (list): Infix tokens.

        Returns:
            list: Postfix tokens.
        """
        precedence = {'+':1, '-':1, '*':2, '/':2}
        output = []
        stack = []
        for token in tokens:
            if token.replace('.', '', 1).isdigit():
                output.append(token)
            elif token in '+-*/':
                while stack and stack[-1] != '(' and precedence[stack[-1]] >= precedence[token]:
                    output.append(stack.pop())
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if not stack or stack.pop() != '(':
                    raise ValueError("Mismatched parentheses.")
            else:
                raise ValueError("Unknown token: '{}'".format(token))
        while stack:
            if stack[-1] == '(':
                raise ValueError("Mismatched parentheses.")
            output.append(stack.pop())
        return output

    def evaluate_postfix(self, tokens):
        """
        Evaluates a postfix expression.

        Args:
            tokens (list): Postfix tokens.

        Returns:
            float: The result of the expression.
        """
        stack = []
        for token in tokens:
            if token.replace('.', '', 1).isdigit():
                stack.append(float(token))
            elif token in '+-*/':
                if len(stack) < 2:
                    raise ValueError("Insufficient values in expression.")
                b = stack.pop()
                a = stack.pop()
                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
                elif token == '/':
                    if b == 0:
                        raise ValueError("Division by zero.")
                    stack.append(a / b)
            else:
                raise ValueError("Unknown token: '{}'".format(token))
        if len(stack) != 1:
            raise ValueError("The user input has too many values.")
        return stack[0]

class ExpressionParser:
    """
    Class for parsing and computing mathematical expressions.
    """

    def compute(self, expression):
        """
        Computes the result of a mathematical expression string.

        Args:
            expression (str): The mathematical expression.

        Returns:
            float: Computed result.
        """
        evaluator = ArithmeticEvaluator()
        return evaluator.evaluate(expression)

class Serializer:
    """
    Class for serializing and deserializing strings.
    """

    def serialize(self, s):
        """
        Serializes a string for safe storage or transmission.

        Escapes special characters.

        Args:
            s (str): The original string.

        Returns:
            str: The serialized string.
        """
        escape_chars = {'\n': '\\n', '\t': '\\t', '\r': '\\r', '\\': '\\\\', '\"': '\\\"', '\'': '\\\''}
        result = []
        for ch in s:
            result.append(escape_chars.get(ch, ch))
        return ''.join(result)

    def deserialize(self, s):
        """
        Deserializes a string from the serialized format.

        Converts escaped sequences back to their original characters.

        Args:
            s (str): The serialized string.

        Returns:
            str: The original string.
        """
        escape_chars = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '\"', '\'': '\''}
        result = []
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                i += 1
                result.append(escape_chars.get(s[i], s[i]))
            else:
                result.append(s[i])
            i += 1
        return ''.join(result)

class StringCompressor:
    """
    Class for compressing and decompressing strings using basic algorithms.
    """

    def compress(self, s):
        """
        Compresses a string using Run-Length Encoding (RLE).

        Args:
            s (str): The original string.

        Returns:
            str: The compressed string.
        """
        if not s:
            return ''
        compressed = []
        count = 1
        prev_char = s[0]
        for ch in s[1:]:
            if ch == prev_char:
                count += 1
            else:
                compressed.append(prev_char + (str(count) if count > 1 else ''))
                prev_char = ch
                count = 1
        compressed.append(prev_char + (str(count) if count > 1 else ''))
        return ''.join(compressed)

    def decompress(self, s):
        """
        Decompresses a string compressed with Run-Length Encoding (RLE).

        Args:
            s (str): The compressed string.

        Returns:
            str: The decompressed string.
        """
        decompressed = []
        i = 0
        while i < len(s):
            ch = s[i]
            i += 1
            count = ''
            while i < len(s) and s[i].isdigit():
                count += s[i]
                i += 1
            decompressed.append(ch * (int(count) if count else 1))
        return ''.join(decompressed)

class StringEncryptor:
    """
    Class for encrypting and decrypting strings using simple cipher algorithms.
    """

    def encrypt(self, s, key):
        """
        Encrypts a string using a simple Caesar cipher.

        Args:
            s (str): The original string.
            key (int): The encryption key.

        Returns:
            str: The encrypted string.
        """
        encrypted = []
        for ch in s:
            if 'a' <= ch <= 'z':
                encrypted.append(chr((ord(ch) - ord('a') + key) % 26 + ord('a')))
            elif 'A' <= ch <= 'Z':
                encrypted.append(chr((ord(ch) - ord('A') + key) % 26 + ord('A')))
            else:
                encrypted.append(ch)
        return ''.join(encrypted)

    def decrypt(self, s, key):
        """
        Decrypts a string encrypted with a simple Caesar cipher.

        Args:
            s (str): The encrypted string.
            key (int): The decryption key.

        Returns:
            str: The decrypted string.
        """
        return self.encrypt(s, -key)

class StringObfuscator:
    """
    Class for obfuscating and deobfuscating strings.
    """

    def obfuscate(self, s):
        """
        Obfuscates a string by reversing and applying a basic transformation.

        Args:
            s (str): The original string.

        Returns:
            str: The obfuscated string.
        """
        return ''.join(chr(ord(ch) ^ 0x42) for ch in s[::-1])

    def deobfuscate(self, s):
        """
        Deobfuscates a string previously obfuscated.

        Args:
            s (str): The obfuscated string.

        Returns:
            str: The original string.
        """
        return ''.join(chr(ord(ch) ^ 0x42) for ch in s)[::-1]

class SyntaxHighlighter:
    """
    Class for preparing strings for syntax highlighting.
    """

    def highlight(self, code_str):
        """
        Marks parts of a code string for syntax highlighting.

        This is a simplified lexer that recognizes keywords, numbers, and identifiers.

        Args:
            code_str (str): The code as a string.

        Returns:
            list: A list of tuples containing (token_type, token_value).
        """
        keywords = {'def', 'class', 'return', 'if', 'else', 'for', 'while', 'in', 'import'}
        tokens = []
        i = 0
        while i < len(code_str):
            if code_str[i].isspace():
                i += 1
                continue
            elif code_str[i].isalpha() or code_str[i] == '_':
                start = i
                while i < len(code_str) and (code_str[i].isalnum() or code_str[i] == '_'):
                    i += 1
                word = code_str[start:i]
                token_type = 'KEYWORD' if word in keywords else 'IDENTIFIER'
                tokens.append((token_type, word))
            elif code_str[i].isdigit():
                start = i
                while i < len(code_str) and code_str[i].isdigit():
                    i += 1
                tokens.append(('NUMBER', code_str[start:i]))
            elif code_str[i] in '()+-*/=:':
                tokens.append(('SYMBOL', code_str[i]))
                i += 1
            elif code_str[i] == '#':
                start = i
                while i < len(code_str) and code_str[i] != '\n':
                    i += 1
                tokens.append(('COMMENT', code_str[start:i]))
            else:
                tokens.append(('UNKNOWN', code_str[i]))
                i += 1
        return tokens

class StringDiffer:
    """
    Class for comparing two strings and finding differences.
    """

    def diff(self, s1, s2):
        """
        Finds the differences between two strings using a simple diff algorithm.

        Args:
            s1 (str): The first string.
            s2 (str): The second string.

        Returns:
            list: A list of tuples indicating (operation, character).
                  Operation can be 'ADD', 'REMOVE', or 'SAME'.
        """
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]

        # Build LCS table
        for i in range(m):
            for j in range(n):
                if s1[i] == s2[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

        # Backtrack to find diff
        diff = []
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i-1] == s2[j-1]:
                diff.append(('SAME', s1[i-1]))
                i -= 1
                j -= 1
            elif dp[i-1][j] >= dp[i][j-1]:
                diff.append(('REMOVE', s1[i-1]))
                i -= 1
            else:
                diff.append(('ADD', s2[j-1]))
                j -= 1
        while i > 0:
            diff.append(('REMOVE', s1[i-1]))
            i -= 1
        while j > 0:
            diff.append(('ADD', s2[j-1]))
            j -= 1
        return diff[::-1]

class VersionComparator:
    """
    Class for comparing version strings.
    """

    def compare(self, v1, v2):
        """
        Compares two version strings.

        Args:
            v1 (str): The first version string.
            v2 (str): The second version string.

        Returns:
            int: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2.
        """
        nums1 = [int(x) for x in v1.split('.')]
        nums2 = [int(x) for x in v2.split('.')]
        len1, len2 = len(nums1), len(nums2)
        for i in range(max(len1, len2)):
            n1 = nums1[i] if i < len1 else 0
            n2 = nums2[i] if i < len2 else 0
            if n1 != n2:
                return 1 if n1 > n2 else -1
        return 0

class EmailValidator:
    """
    Class for parsing and validating email addresses.
    """

    def is_valid_email(self, email):
        """
        Validates an email address.

        Args:
            email (str): The email address to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not email or '@' not in email:
            return False
        local_part, domain = email.rsplit('@', 1)
        if not local_part or not domain:
            return False
        if '..' in email:
            return False
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'*+-/=?^_`{|}~.")
        if not set(local_part).issubset(allowed_chars):
            return False
        if '.' not in domain:
            return False
        return True

class URLParser:
    """
    Class for parsing and validating URLs.
    """

    def parse_url(self, url):
        """
        Parses a URL and extracts its components.

        Args:
            url (str): The URL to parse.

        Returns:
            dict: A dictionary with URL components.
        """
        scheme = ''
        netloc = ''
        path = ''
        query = ''
        fragment = ''
        i = url.find('://')
        if i != -1:
            scheme = url[:i]
            url = url[i+3:]
        else:
            scheme = 'http'  # Default scheme
        i = url.find('/')
        if i != -1:
            netloc = url[:i]
            url = url[i:]
            i = url.find('#')
            if i != -1:
                fragment = url[i+1:]
                url = url[:i]
            i = url.find('?')
            if i != -1:
                query = url[i+1:]
                path = url[:i]
            else:
                path = url
        else:
            netloc = url
        return {
            'scheme': scheme,
            'netloc': netloc,
            'path': path,
            'query': query,
            'fragment': fragment
        }

    def is_valid_url(self, url):
        """
        Validates the URL format.

        Args:
            url (str): The URL to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        components = self.parse_url(url)
        if not components['scheme'] or not components['netloc']:
            return False
        if '.' not in components['netloc']:
            return False
        return True

class IPAddressParser:
    """
    Class for parsing and validating IP addresses.
    """

    def parse_ipv4(self, ip):
        """
        Parses an IPv4 address.

        Args:
            ip (str): The IPv4 address.

        Returns:
            list: A list of four integers representing the IP address.

        Raises:
            ValueError: If the IP address is invalid.
        """
        parts = ip.split('.')
        if len(parts) != 4:
            raise ValueError("Invalid IPv4 address.")
        nums = []
        for part in parts:
            if not part.isdigit():
                raise ValueError("Invalid IPv4 address segment: '{}'".format(part))
            num = int(part)
            if num < 0 or num > 255:
                raise ValueError("IPv4 address segment out of range: {}".format(num))
            nums.append(num)
        return nums

    def is_valid_ipv4(self, ip):
        """
        Validates an IPv4 address.

        Args:
            ip (str): The IPv4 address.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            self.parse_ipv4(ip)
            return True
        except ValueError:
            return False

class QueryStringParser:
    """
    Class for encoding and decoding URL query parameters.
    """

    def decode_query(self, query):
        """
        Decodes URL query parameters into a dictionary.

        Args:
            query (str): The query string.

        Returns:
            dict: A dictionary of key-value pairs.
        """
        params = {}
        pairs = query.split('&')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                params[self.url_decode(key)] = self.url_decode(value)
            else:
                params[self.url_decode(pair)] = ''
        return params

    def encode_query(self, params):
        """
        Encodes a dictionary into URL query parameters.

        Args:
            params (dict): Dictionary of key-value pairs.

        Returns:
            str: The encoded query string.
        """
        pairs = []
        for key, value in params.items():
            pairs.append(self.url_encode(key) + '=' + self.url_encode(value))
        return '&'.join(pairs)

    def url_encode(self, s):
        """
        Encodes a string for use in a URL.

        Args:
            s (str): The string to encode.

        Returns:
            str: The encoded string.
        """
        result = []
        for ch in s:
            if ch.isalnum() or ch in '-_.~':
                result.append(ch)
            else:
                result.append('%{:02X}'.format(ord(ch)))
        return ''.join(result)

    def url_decode(self, s):
        """
        Decodes a URL-encoded string.

        Args:
            s (str): The encoded string.

        Returns:
            str: The decoded string.
        """
        result = []
        i = 0
        while i < len(s):
            if s[i] == '%' and i + 2 < len(s) and s[i+1:i+3].isalnum():
                hex_value = s[i+1:i+3]
                result.append(chr(int(hex_value, 16)))
                i += 3
            else:
                result.append(s[i])
                i += 1
        return ''.join(result)

class HTMLStripper:
    """
    Class for removing HTML tags from strings.
    """

    def strip_tags(self, html):
        """
        Removes HTML tags from a string.

        Args:
            html (str): The HTML string.

        Returns:
            str: The plain text string.
        """
        result = []
        in_tag = False
        for ch in html:
            if ch == '<':
                in_tag = True
            elif ch == '>':
                in_tag = False
            elif not in_tag:
                result.append(ch)
        return ''.join(result)

class CodeSyntaxParser:
    """
    Class for parsing code strings.
    """

    def parse_code(self, code_str):
        """
        Analyzes code snippets and returns a structured representation.

        Args:
            code_str (str): The code string.

        Returns:
            list: A list of code tokens.
        """
        # Reusing highlight function for demo purposes
        highlighter = SyntaxHighlighter()
        return highlighter.highlight(code_str)

class NLPTokens:
    """
    Class for tokenizing text for NLP tasks.
    """

    def tokenize(self, text):
        """
        Tokenizes text into words considering punctuation and special cases.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: A list of tokens.
        """
        tokens = []
        current_token = []
        for ch in text:
            if ch.isalnum():
                current_token.append(ch)
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                if ch.strip():
                    tokens.append(ch)
        if current_token:
            tokens.append(''.join(current_token))
        return tokens

class SentimentPreprocessor:
    """
    Class for preprocessing text for sentiment analysis.
    """

    def preprocess(self, text):
        """
        Preprocesses text by converting to lowercase and removing punctuation.

        Args:
            text (str): The original text.

        Returns:
            str: The preprocessed text.
        """
        lowered = text.lower()
        cleaned = ''.join(ch for ch in lowered if ch.isalnum() or ch.isspace())
        return cleaned

class EntityExtractor:
    """
    Class for extracting named entities from text.
    """

    def extract_entities(self, text):
        """
        Extracts capitalized words as named entities.

        Args:
            text (str): The text to analyze.

        Returns:
            list: A list of named entities.
        """
        tokens = text.split()
        entities = [token for token in tokens if token.istitle()]
        return entities

class TextSummarizer:
    """
    Class for preparing text for summarization.
    """

    def prepare(self, text):
        """
        Prepares text by removing stop words.

        Args:
            text (str): The original text.

        Returns:
            str: The text without stop words.
        """
        stop_words = {'the', 'and', 'is', 'in', 'at', 'of', 'a', 'an'}
        tokens = text.lower().split()
        filtered = [token for token in tokens if token not in stop_words]
        return ' '.join(filtered)

class EscapeSequenceConverter:
    """
    Class for converting escape sequences in strings.
    """

    def escape_sequences_to_characters(self, s):
        """
        Converts escape sequences like '\\n' to actual characters.

        Args:
            s (str): The string with escape sequences.

        Returns:
            str: The string with actual characters.
        """
        serializer = Serializer()
        return serializer.deserialize(s)

    def characters_to_escape_sequences(self, s):
        """
        Replaces certain characters with their escape sequence equivalents.

        Args:
            s (str): The string with actual characters.

        Returns:
            str: The string with escape sequences.
        """
        serializer = Serializer()
        return serializer.serialize(s)

class SentenceRearranger:
    """
    Class for parsing and rearranging sentence structures.
    """

    def rearrange(self, sentence):
        """
        Rearranges a sentence to a specific structure.

        For example, moves the last word to the beginning.

        Args:
            sentence (str): The original sentence.

        Returns:
            str: The rearranged sentence.
        """
        words = sentence.split()
        if not words:
            return sentence
        rearranged = [words[-1]] + words[:-1]
        return ' '.join(rearranged)

class LineSorter:
    """
    Class for sorting lines in a multiline string.
    """

    def sort_lines(self, s):
        """
        Sorts lines in a multiline string alphabetically.

        Args:
            s (str): The multiline string.

        Returns:
            str: The string with sorted lines.
        """
        lines = s.split('\n')
        lines.sort()
        return '\n'.join(lines)

class DiacriticRemover:
    """
    Class for removing accents and diacritics from characters.
    """

    def remove_accents(self, s):
        """
        Removes accents from characters by mapping them to ASCII equivalents.

        Args:
            s (str): The original string.

        Returns:
            str: The string without accents.
        """
        mapping = {
            'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u',
            'Á':'A', 'É':'E', 'Í':'I', 'Ó':'O', 'Ú':'U',
            'à':'a', 'è':'e', 'ì':'i', 'ò':'o', 'ù':'u',
            'À':'A', 'È':'E', 'Ì':'I', 'Ò':'O', 'Ù':'U',
            'ä':'a', 'ë':'e', 'ï':'i', 'ö':'o', 'ü':'u',
            'Ä':'A', 'Ë':'E', 'Ï':'I', 'Ö':'O', 'Ü':'U',
            'â':'a', 'ê':'e', 'î':'i', 'ô':'o', 'û':'u',
            'Â':'A', 'Ê':'E', 'Î':'I', 'Ô':'O', 'Û':'U',
            'ç':'c', 'Ç':'C', 'ñ':'n', 'Ñ':'N'
        }
        return ''.join(mapping.get(ch, ch) for ch in s)

class CustomCollator:
    """
    Class for applying custom collation rules on strings.
    """

    def sort_strings(self, strings, order):
        """
        Sorts strings using a custom character order.

        Args:
            strings (list): A list of strings to sort.
            order (str): A string representing the custom order.

        Returns:
            list: The sorted list of strings.
        """
        order_map = {ch: idx for idx, ch in enumerate(order)}
        def custom_key(s):
            return [order_map.get(ch, -1) for ch in s]
        return sorted(strings, key=custom_key)

# Dummy test cases
if __name__ == "__main__":
    # StringParser tests
    parser = StringParser()
    print(parser.parse_int("  -123 "))  # Output: -123
    print(parser.parse_float("45.67"))  # Output: 45.67
    print(parser.int_to_string(-789))   # Output: "-789"
    print(parser.float_to_string(3.14159, precision=4))  # Output: "3.1415"

    # ArithmeticEvaluator tests
    evaluator = ArithmeticEvaluator()
    print(evaluator.evaluate("3 + 4 * 5 / 2"))  # Output: 13.0

    # ExpressionParser tests
    expr_parser = ExpressionParser()
    print(expr_parser.compute("(2 + 3) * (7 - 4)"))  # Output: 15.0

    # Serializer tests
    serializer = Serializer()
    serialized = serializer.serialize("Hello\nWorld\t!")
    print(serialized)  # Output: "Hello\\nWorld\\t!"
    deserialized = serializer.deserialize(serialized)
    print(deserialized)  # Output: "Hello\nWorld\t!"

    # StringCompressor tests
    compressor = StringCompressor()
    compressed = compressor.compress("aaabccccdd")
    print(compressed)  # Output: "a3bc4d2"
    decompressed = compressor.decompress(compressed)
    print(decompressed)  # Output: "aaabccccdd"

    # StringEncryptor tests
    encryptor = StringEncryptor()
    encrypted = encryptor.encrypt("Hello World!", 3)
    print(encrypted)  # Output: "Khoor Zruog!"
    decrypted = encryptor.decrypt(encrypted, 3)
    print(decrypted)  # Output: "Hello World!"

    # StringObfuscator tests
    obfuscator = StringObfuscator()
    obfuscated = obfuscator.obfuscate("SecretMessage")
    print(obfuscated)
    deobfuscated = obfuscator.deobfuscate(obfuscated)
    print(deobfuscated)  # Output: "SecretMessage"

    # SyntaxHighlighter tests
    highlighter = SyntaxHighlighter()
    tokens = highlighter.highlight("def foo():\n    return 42")
    print(tokens)  # Output: tokenized code

    # StringDiffer tests
    differ = StringDiffer()
    differences = differ.diff("abcdef", "abcxyz")
    print(differences)  # Output: list of differences

    # VersionComparator tests
    comparator = VersionComparator()
    compare_result = comparator.compare("1.2.3", "1.2.10")
    print(compare_result)  # Output: -1

    # EmailValidator tests
    email_validator = EmailValidator()
    is_valid = email_validator.is_valid_email("user@example.com")
    print(is_valid)  # Output: True

    # URLParser tests
    url_parser = URLParser()
    url_components = url_parser.parse_url("https://www.example.com/path?query=value#fragment")
    print(url_components)

    # IPAddressParser tests
    ip_parser = IPAddressParser()
    is_valid_ip = ip_parser.is_valid_ipv4("192.168.1.1")
    print(is_valid_ip)  # Output: True

    # QueryStringParser tests
    query_parser = QueryStringParser()
    params = query_parser.decode_query("name=John%20Doe&age=30")
    print(params)
    query_string = query_parser.encode_query({'name': 'John Doe', 'age': '30'})
    print(query_string)

    # HTMLStripper tests
    stripper = HTMLStripper()
    plain_text = stripper.strip_tags("<p>Hello <b>World</b>!</p>")
    print(plain_text)  # Output: "Hello World!"

    # CodeSyntaxParser tests
    code_parser = CodeSyntaxParser()
    code_tokens = code_parser.parse_code("for i in range(5): print(i)")
    print(code_tokens)

    # NLPTokens tests
    tokenizer = NLPTokens()
    tokens = tokenizer.tokenize("Hello, world! It's a beautiful day.")
    print(tokens)

    # SentimentPreprocessor tests
    preprocessor = SentimentPreprocessor()
    preprocessed_text = preprocessor.preprocess("I love sunny days!")
    print(preprocessed_text)  # Output: "i love sunny days"

    # EntityExtractor tests
    extractor = EntityExtractor()
    entities = extractor.extract_entities("Alice went to Wonderland.")
    print(entities)  # Output: ['Alice', 'Wonderland.']

    # TextSummarizer tests
    summarizer = TextSummarizer()
    summary_ready_text = summarizer.prepare("The quick brown fox jumps over the lazy dog.")
    print(summary_ready_text)  # Output: "quick brown fox jumps over lazy dog"

    # EscapeSequenceConverter tests
    esc_converter = EscapeSequenceConverter()
    actual_chars = esc_converter.escape_sequences_to_characters("Line1\\nLine2\\tTabbed")
    print(actual_chars)  # Output: "Line1\nLine2\tTabbed"
    escaped_sequences = esc_converter.characters_to_escape_sequences("Line1\nLine2\tTabbed")
    print(escaped_sequences)

    # SentenceRearranger tests
    rearranger = SentenceRearranger()
    rearranged_sentence = rearranger.rearrange("The sky is blue")
    print(rearranged_sentence)  # Output: "blue The sky is"

    # LineSorter tests
    sorter = LineSorter()
    sorted_lines = sorter.sort_lines("banana\napple\ncherry")
    print(sorted_lines)  # Output: "apple\nbanana\ncherry"

    # DiacriticRemover tests
    remover = DiacriticRemover()
    normalized_text = remover.remove_accents("Cliché naïve façade résumé")
    print(normalized_text)  # Output: "Cliche naive facade resume"

    # CustomCollator tests
    collator = CustomCollator()
    custom_sorted = collator.sort_strings(['apple', 'banana', 'cherry'], 'zyxwvutsrqponmlkjihgfedcba')
    print(custom_sorted)  # Output: sorted in reverse alphabetical order