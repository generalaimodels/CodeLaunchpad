class StringOperations:
    """
    A class that implements various string operations without using standard library modules.
    """

    def tokenize(self, text):
        """
        Breaks the given text into words or tokens based on spaces.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of words/tokens.
        """
        tokens = []
        word = ''
        for char in text:
            if char == ' ' or char == '\n' or char == '\t':
                if word != '':
                    tokens.append(word)
                    word = ''
            else:
                word += char
        if word != '':
            tokens.append(word)
        return tokens

    def join_strings(self, strings_list, delimiter=''):
        """
        Combines a list of strings into one string with an optional delimiter.

        Args:
            strings_list (List[str]): The list of strings to combine.
            delimiter (str): The delimiter to use between strings.

        Returns:
            str: The combined string.
        """
        result = ''
        for i in range(len(strings_list)):
            result += strings_list[i]
            if i != len(strings_list) - 1:
                result += delimiter
        return result

    def repeat_string(self, s, n):
        """
        Repeats a string a given number of times.

        Args:
            s (str): The string to repeat.
            n (int): Number of times to repeat.

        Returns:
            str: The repeated string.
        """
        result = ''
        for _ in range(n):
            result += s
        return result

    def replicate_string(self, s, n):
        """
        Replicates a string using multiplication operator.

        Args:
            s (str): The string to replicate.
            n (int): The number of times to replicate.

        Returns:
            str: The replicated string.
        """
        # Since we can't use the * operator, we use the same as repeat_string
        return self.repeat_string(s, n)

    def pad_string(self, s, total_length, pad_char=' '):
        """
        Pads the string with pad_char to reach the total_length.

        Args:
            s (str): The original string.
            total_length (int): Desired total length after padding.
            pad_char (str): Character to use for padding.

        Returns:
            str: The padded string, evenly padded on both sides if possible.
        """
        current_length = len(s)
        if current_length >= total_length:
            return s
        pad_total = total_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        result = self.repeat_string(pad_char, pad_left) + s + self.repeat_string(pad_char, pad_right)
        return result

    def pad_start(self, s, total_length, pad_char=' '):
        """
        Pads the beginning of the string to reach the total_length.

        Args:
            s (str): The original string.
            total_length (int): Desired total length after padding.
            pad_char (str): Character to use for padding.

        Returns:
            str: The padded string.
        """
        current_length = len(s)
        if current_length >= total_length:
            return s
        pad_total = total_length - current_length
        result = self.repeat_string(pad_char, pad_total) + s
        return result

    def pad_end(self, s, total_length, pad_char=' '):
        """
        Pads the end of the string to reach the total_length.

        Args:
            s (str): The original string.
            total_length (int): Desired total length after padding.
            pad_char (str): Character to use for padding.

        Returns:
            str: The padded string.
        """
        current_length = len(s)
        if current_length >= total_length:
            return s
        pad_total = total_length - current_length
        result = s + self.repeat_string(pad_char, pad_total)
        return result

    def substring_insert(self, s, substring, index):
        """
        Inserts a substring into a string at a specific index.

        Args:
            s (str): The original string.
            substring (str): The substring to insert.
            index (int): The index at which to insert.

        Returns:
            str: The string after insertion.
        """
        if index < 0:
            index = 0
        elif index > len(s):
            index = len(s)
        prefix = ''
        suffix = ''
        for i in range(len(s)):
            if i < index:
                prefix += s[i]
            else:
                suffix += s[i]
        return prefix + substring + suffix

    def substring_delete(self, s, start_index, length):
        """
        Removes a section from a string.

        Args:
            s (str): The original string.
            start_index (int): The starting index of the section to remove.
            length (int): The length of the section to remove.

        Returns:
            str: The string after deletion.
        """
        if start_index < 0:
            start_index = 0
        if length < 0:
            return s
        prefix = ''
        suffix = ''
        for i in range(len(s)):
            if i < start_index:
                prefix += s[i]
            elif i >= start_index + length:
                suffix += s[i]
            else:
                continue  # Skip characters to delete
        return prefix + suffix

    def substring_replace(self, s, old_substring, new_substring):
        """
        Replaces the first occurrence of old_substring with new_substring.

        Args:
            s (str): The original string.
            old_substring (str): The substring to replace.
            new_substring (str): The substring to insert.

        Returns:
            str: The string after replacement.
        """
        index = self.find_substring(s, old_substring)
        if index == -1:
            return s
        result = self.substring_delete(s, index, len(old_substring))
        result = self.substring_insert(result, new_substring, index)
        return result

    def global_substring_replace(self, s, old_substring, new_substring):
        """
        Replaces all occurrences of old_substring with new_substring.

        Args:
            s (str): The original string.
            old_substring (str): The substring to replace.
            new_substring (str): The substring to insert.

        Returns:
            str: The string after replacement.
        """
        result = ''
        i = 0
        while i < len(s):
            if self.starts_with(s, old_substring, i):
                result += new_substring
                i += len(old_substring)
            else:
                result += s[i]
                i += 1
        return result

    def case_insensitive_replace(self, s, old_substring, new_substring):
        """
        Replaces all occurrences of old_substring with new_substring, regardless of case.

        Args:
            s (str): The original string.
            old_substring (str): The substring to replace.
            new_substring (str): The substring to insert.

        Returns:
            str: The string after replacement.
        """
        s_lower = self.to_lower(s)
        old_substring_lower = self.to_lower(old_substring)
        result = ''
        i = 0
        while i < len(s):
            if self.starts_with(s_lower, old_substring_lower, i):
                result += new_substring
                i += len(old_substring)
            else:
                result += s[i]
                i += 1
        return result

    def swap_characters(self, s, index1, index2):
        """
        Swaps the characters at two indices in the string.

        Args:
            s (str): The original string.
            index1 (int): The index of the first character.
            index2 (int): The index of the second character.

        Returns:
            str: The string after swapping characters.
        """
        if index1 < 0 or index1 >= len(s) or index2 < 0 or index2 >= len(s):
            return s  # Indices out of bounds
        result_chars = []
        for i in range(len(s)):
            if i == index1:
                result_chars.append(s[index2])
            elif i == index2:
                result_chars.append(s[index1])
            else:
                result_chars.append(s[i])
        result = ''
        for char in result_chars:
            result += char
        return result

    def reverse_string(self, s):
        """
        Reverses the string.

        Args:
            s (str): The original string.

        Returns:
            str: The reversed string.
        """
        result = ''
        for i in range(len(s) - 1, -1, -1):
            result += s[i]
        return result

    def sort_characters(self, s):
        """
        Sorts the characters in the string alphabetically.

        Args:
            s (str): The original string.

        Returns:
            str: The string with sorted characters.
        """
        char_list = []
        for char in s:
            char_list.append(char)
        self.quick_sort(char_list, 0, len(char_list) - 1)
        result = ''
        for char in char_list:
            result += char
        return result

    def shuffle_characters(self, s):
        """
        Randomly rearranges the characters in the string.

        Args:
            s (str): The original string.

        Returns:
            str: The string with shuffled characters.
        """
        # Since we cannot use random module, we need to implement a simple pseudo-random generator
        # We'll use a simple shuffle algorithm with a seed
        char_list = []
        for char in s:
            char_list.append(char)
        seed = len(s)  # Simple seed
        for i in range(len(char_list) - 1, 0, -1):
            # Simple pseudo-random index selection
            seed = (seed * 9301 + 49297) % 233280
            rnd = seed / 233280.0
            j = int(rnd * (i + 1))
            # Swap char_list[i] with char_list[j]
            temp = char_list[i]
            char_list[i] = char_list[j]
            char_list[j] = temp
        result = ''
        for char in char_list:
            result += char
        return result

    def remove_duplicate_characters(self, s):
        """
        Removes duplicate characters from the string.

        Args:
            s (str): The original string.

        Returns:
            str: The string without duplicate characters.
        """
        seen = {}
        result = ''
        for char in s:
            if char not in seen:
                seen[char] = True
                result += char
        return result

    def remove_duplicate_words(self, s):
        """
        Removes duplicate words from the sentence.

        Args:
            s (str): The original sentence.

        Returns:
            str: The sentence without duplicate words.
        """
        words = self.tokenize(s)
        seen = {}
        result_words = []
        for word in words:
            if word not in seen:
                seen[word] = True
                result_words.append(word)
        result = self.join_strings(result_words, ' ')
        return result

    def split_sentence(self, s):
        """
        Splits a sentence into words using spaces and punctuation as delimiters.

        Args:
            s (str): The original sentence.

        Returns:
            List[str]: The list of words.
        """
        tokens = []
        word = ''
        for char in s:
            if self.is_alpha_numeric(char):
                word += char
            else:
                if word != '':
                    tokens.append(word)
                    word = ''
        if word != '':
            tokens.append(word)
        return tokens

    def count_words(self, s):
        """
        Counts the number of words in a sentence.

        Args:
            s (str): The sentence.

        Returns:
            int: The word count.
        """
        words = self.split_sentence(s)
        return len(words)

    def count_substring_occurrences(self, s, substring):
        """
        Counts how many times a substring appears in a string.

        Args:
            s (str): The original string.
            substring (str): The substring to count.

        Returns:
            int: The number of occurrences.
        """
        count = 0
        index = 0
        while index <= len(s) - len(substring):
            if self.starts_with(s, substring, index):
                count += 1
                index += len(substring)
            else:
                index += 1
        return count

    # Helper methods

    def find_substring(self, s, substring):
        """
        Finds the index of the first occurrence of substring in s.

        Args:
            s (str): The original string.
            substring (str): The substring to find.

        Returns:
            int: The index of the first occurrence, or -1 if not found.
        """
        for i in range(len(s) - len(substring) + 1):
            if self.starts_with(s, substring, i):
                return i
        return -1

    def starts_with(self, s, substring, index):
        """
        Checks if s starts with substring at the given index.

        Args:
            s (str): The original string.
            substring (str): The substring to check.
            index (int): The index to start checking.

        Returns:
            bool: True if s starts with substring at index, False otherwise.
        """
        if index + len(substring) > len(s):
            return False
        for i in range(len(substring)):
            if s[index + i] != substring[i]:
                return False
        return True

    def is_alpha_numeric(self, char):
        """
        Checks if a character is alphanumeric.

        Args:
            char (str): The character to check.

        Returns:
            bool: True if alphanumeric, False otherwise.
        """
        ascii_code = ord(char)
        if (ascii_code >= ord('a') and ascii_code <= ord('z')) or \
           (ascii_code >= ord('A') and ascii_code <= ord('Z')) or \
           (ascii_code >= ord('0') and ascii_code <= ord('9')):
            return True
        return False

    def to_lower(self, s):
        """
        Converts a string to lowercase.

        Args:
            s (str): The original string.

        Returns:
            str: The lowercase string.
        """
        result = ''
        for char in s:
            ascii_code = ord(char)
            if ascii_code >= ord('A') and ascii_code <= ord('Z'):
                result += chr(ascii_code + 32)
            else:
                result += char
        return result

    def quick_sort(self, arr, low, high):
        """
        Performs quick sort on an array of characters.

        Args:
            arr (List[str]): The array to sort.
            low (int): The starting index.
            high (int): The ending index.
        """
        if low < high:
            pi = self.partition(arr, low, high)
            self.quick_sort(arr, low, pi - 1)
            self.quick_sort(arr, pi + 1, high)

    def partition(self, arr, low, high):
        """
        Helper function for quick sort. Partitions the array.

        Args:
            arr (List[str]): The array to partition.
            low (int): The starting index.
            high (int): The ending index.

        Returns:
            int: The partition index.
        """
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                # Swap arr[i] and arr[j]
                arr[i], arr[j] = arr[j], arr[i]
        # Swap arr[i+1] and arr[high] (or pivot)
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1


# Dummy test cases
if __name__ == "__main__":
    string_ops = StringOperations()

    # Test tokenize
    text = "Hello, world! Welcome to OpenAI."
    tokens = string_ops.tokenize(text)
    print("Tokenize:", tokens)

    # Test join_strings
    strings_list = ['Hello', 'world', 'OpenAI']
    joined_string = string_ops.join_strings(strings_list, ' ')
    print("Join Strings:", joined_string)

    # Test repeat_string
    repeated_string = string_ops.repeat_string('abc', 3)
    print("Repeated String:", repeated_string)

    # Test pad_string
    padded_string = string_ops.pad_string('abc', 10, '*')
    print("Padded String:", padded_string)

    # Test pad_start
    pad_start_string = string_ops.pad_start('abc', 6, '0')
    print("Pad Start:", pad_start_string)

    # Test pad_end
    pad_end_string = string_ops.pad_end('abc', 6, '0')
    print("Pad End:", pad_end_string)

    # Test substring_insert
    insert_string = string_ops.substring_insert('hello world', ' beautiful', 5)
    print("Substring Insert:", insert_string)

    # Test substring_delete
    delete_string = string_ops.substring_delete('hello beautiful world', 6, 10)
    print("Substring Delete:", delete_string)

    # Test substring_replace
    replace_string = string_ops.substring_replace('hello world', 'world', 'Python')
    print("Substring Replace:", replace_string)

    # Test global_substring_replace
    global_replace_string = string_ops.global_substring_replace('abc abc abc', 'abc', 'xyz')
    print("Global Substring Replace:", global_replace_string)

    # Test case_insensitive_replace
    case_insensitive_string = string_ops.case_insensitive_replace('Hello hello HeLLo', 'hello', 'hi')
    print("Case Insensitive Replace:", case_insensitive_string)

    # Test swap_characters
    swap_string = string_ops.swap_characters('abcd', 1, 3)
    print("Swap Characters:", swap_string)

    # Test reverse_string
    reverse_string = string_ops.reverse_string('abcdef')
    print("Reverse String:", reverse_string)

    # Test sort_characters
    sorted_string = string_ops.sort_characters('dcba')
    print("Sort Characters:", sorted_string)

    # Test shuffle_characters
    shuffled_string = string_ops.shuffle_characters('abcdef')
    print("Shuffle Characters:", shuffled_string)

    # Test remove_duplicate_characters
    unique_chars_string = string_ops.remove_duplicate_characters('aabbccdd')
    print("Remove Duplicate Characters:", unique_chars_string)

    # Test remove_duplicate_words
    unique_words_string = string_ops.remove_duplicate_words('hello hello world world')
    print("Remove Duplicate Words:", unique_words_string)

    # Test split_sentence
    words_list = string_ops.split_sentence('Hello, world! Welcome to OpenAI.')
    print("Split Sentence:", words_list)

    # Test count_words
    word_count = string_ops.count_words('Hello, world! Welcome to OpenAI.')
    print("Count Words:", word_count)

    # Test count_substring_occurrences
    substring_count = string_ops.count_substring_occurrences('abcabcabc', 'abc')
    print("Count Substring Occurrences:", substring_count)