class StringOperations:
    """
    A class that encapsulates various string operations including phonetic encodings,
    word manipulations, case conversions, tokenizations, analytics, and more.
    """

    def soundex(self, word):
        """
        Implements the Soundex algorithm for phonetic encoding of English words.

        :param word: The input word to encode.
        :return: The Soundex code for the input word.
        """
        if not word:
            return ""

        # Convert to uppercase
        word = word.upper()

        # Soundex mapping
        mappings = {
            'A': '', 'E': '', 'I': '', 'O': '', 'U': '', 'Y': '', 'H': '', 'W': '',
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }

        # Save the first letter
        first_letter = word[0]

        # Encode the rest of the letters
        encoded = first_letter
        prev_digit = mappings.get(first_letter, '')
        for char in word[1:]:
            digit = mappings.get(char, '')
            if digit != prev_digit:
                encoded += digit
                prev_digit = digit

        # Remove zeros and pad/truncate to length 4
        encoded = ''.join(filter(None, [char for char in encoded]))
        encoded = (encoded + '000')[:4]

        return encoded

    def metaphone(self, word):
        """
        Implements the basic Metaphone algorithm for phonetic encoding.

        :param word: The input word to encode.
        :return: The Metaphone code for the input word.
        """
        if not word:
            return ""

        word = word.lower()
        vowels = "aeiou"

        result = ""
        index = 0
        length = len(word)

        while index < length:
            char = word[index]
            if char in vowels:
                if index == 0:
                    result += char.upper()
                index += 1
            elif char == 'b':
                result += 'B'
                index += 1 if word[index - 1:index + 1] != 'mb' else 2
            elif char in ('c', 'k', 'q'):
                result += 'K'
                index += 1
            elif char == 'd':
                if word[index:index + 2] == 'dg' and word[index + 2:index + 3] in vowels:
                    result += 'J'
                    index += 2
                else:
                    result += 'T'
                    index += 1
            elif char == 'f':
                result += 'F'
                index += 1
            elif char in ('g', 'j'):
                if word[index:index + 2] == 'gn' or word[index:index + 4] == 'gned':
                    index += 1
                elif word[index + 1:index + 2] == 'h':
                    if index > 0 and word[index - 1] in vowels:
                        result += 'J'
                        index += 2
                    else:
                        index += 2
                elif char == 'g' and word[index + 1:index + 2] in ('e', 'i', 'y'):
                    result += 'J'
                    index += 1
                else:
                    result += 'K'
                    index += 1
            elif char == 'h':
                if index == 0 or word[index - 1] not in vowels:
                    result += 'H'
                    index += 1
                else:
                    index += 1
            elif char == 'l':
                result += 'L'
                index += 1
            elif char == 'm':
                result += 'M'
                index += 1
            elif char == 'n':
                result += 'N'
                index += 1
            elif char == 'p':
                if word[index + 1:index + 2] == 'h':
                    result += 'F'
                    index += 2
                else:
                    result += 'P'
                    index += 1
            elif char == 'r':
                result += 'R'
                index += 1
            elif char == 's':
                if word[index:index + 2] == 'sh' or word[index:index + 3] == 'sio' or word[index:index + 3] == 'sia':
                    result += 'X'
                    index += 2
                else:
                    result += 'S'
                    index += 1
            elif char == 't':
                if word[index:index + 2] == 'th':
                    result += '0'
                    index += 2
                elif word[index:index + 3] == 'tio' or word[index:index + 3] == 'tia':
                    result += 'X'
                    index += 3
                else:
                    result += 'T'
                    index += 1
            elif char == 'v':
                result += 'F'
                index += 1
            elif char == 'w' or char == 'y':
                if index + 1 < length and word[index + 1] in vowels:
                    result += char.upper()
                index += 1
            elif char == 'x':
                result += 'KS'
                index += 1
            elif char == 'z':
                result += 'S'
                index += 1
            else:
                index += 1

        return result

    def porter_stem(self, word):
        """
        Implements the Porter Stemming algorithm to reduce words to their stems.

        :param word: The input word to stem.
        :return: The stemmed version of the word.
        """
        if not word:
            return ""

        word = word.lower()
        suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']

        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]
        return word

    def pluralize(self, word):
        """
        Converts a singular word to its plural form.

        :param word: The singular word.
        :return: The plural form of the word.
        """
        if not word:
            return ""

        if word.endswith('y') and word[-2] not in 'aeiou':
            return word[:-1] + 'ies'
        elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return word + 'es'
        elif word.endswith('f'):
            return word[:-1] + 'ves'
        elif word.endswith('fe'):
            return word[:-2] + 'ves'
        else:
            return word + 's'

    def singularize(self, word):
        """
        Converts a plural word to its singular form.

        :param word: The plural word.
        :return: The singular form of the word.
        """
        if not word:
            return ""

        if word.endswith('ies') and len(word) > 3:
            return word[:-3] + 'y'
        elif word.endswith('ves') and len(word) > 3:
            return word[:-3] + 'f'
        elif word.endswith('s') and not word.endswith('ss'):
            return word[:-1]
        else:
            return word

    def to_camel_case(self, text):
        """
        Converts a string to camelCase.

        :param text: The input text.
        :return: The camelCase version of the text.
        """
        words = self.word_tokenize(text)
        if not words:
            return ""

        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

    def to_pascal_case(self, text):
        """
        Converts a string to PascalCase.

        :param text: The input text.
        :return: The PascalCase version of the text.
        """
        words = self.word_tokenize(text)
        return ''.join(word.capitalize() for word in words)

    def to_snake_case(self, text):
        """
        Converts a string to snake_case.

        :param text: The input text.
        :return: The snake_case version of the text.
        """
        words = self.word_tokenize(text)
        return '_'.join(words).lower()

    def to_kebab_case(self, text):
        """
        Converts a string to kebab-case.

        :param text: The input text.
        :return: The kebab-case version of the text.
        """
        words = self.word_tokenize(text)
        return '-'.join(words).lower()

    def sentence_split(self, text):
        """
        Splits a text into sentences.

        :param text: The input text.
        :return: A list of sentences.
        """
        sentences = []
        sentence = ''
        for char in text:
            sentence += char
            if char in '.!?':
                if sentence.strip():
                    sentences.append(sentence.strip())
                    sentence = ''
        if sentence.strip():
            sentences.append(sentence.strip())
        return sentences

    def word_tokenize(self, text):
        """
        Splits a sentence into words.

        :param text: The input text.
        :return: A list of words.
        """
        words = []
        word = ''
        for char in text:
            if char.isalnum():
                word += char
            elif word:
                words.append(word)
                word = ''
        if word:
            words.append(word)
        return words

    def count_sentences(self, text):
        """
        Counts the number of sentences in the text.

        :param text: The input text.
        :return: The number of sentences.
        """
        sentences = self.sentence_split(text)
        return len(sentences)

    def count_lines(self, text):
        """
        Counts the number of newline-separated lines in the text.

        :param text: The input text.
        :return: The number of lines.
        """
        return text.count('\n') + 1 if text else 0

    def average_word_length(self, text):
        """
        Calculates the average word length in the text.

        :param text: The input text.
        :return: The average word length.
        """
        words = self.word_tokenize(text)
        if not words:
            return 0
        total_length = sum(len(word) for word in words)
        return total_length / len(words)

    def average_sentence_length(self, text):
        """
        Calculates the average sentence length in words.

        :param text: The input text.
        :return: The average sentence length.
        """
        sentences = self.sentence_split(text)
        if not sentences:
            return 0
        total_words = sum(len(self.word_tokenize(sentence)) for sentence in sentences)
        return total_words / len(sentences)

    def longest_word(self, text):
        """
        Identifies the longest word in the text.

        :param text: The input text.
        :return: The longest word.
        """
        words = self.word_tokenize(text)
        if not words:
            return ""
        return max(words, key=len)

    def shortest_word(self, text):
        """
        Identifies the shortest word in the text.

        :param text: The input text.
        :return: The shortest word.
        """
        words = self.word_tokenize(text)
        if not words:
            return ""
        return min(words, key=len)

    def sort_words_alphabetically(self, text):
        """
        Sorts the words in the text alphabetically.

        :param text: The input text.
        :return: A list of sorted words.
        """
        words = self.word_tokenize(text)
        return sorted(words, key=lambda s: s.lower())

    def remove_stop_words(self, text, stop_words=None):
        """
        Removes common stop words from the text.

        :param text: The input text.
        :param stop_words: A set of stop words to remove.
        :return: The text without stop words.
        """
        if stop_words is None:
            stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an'}
        words = self.word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    def extract_keywords(self, text, top_n=5):
        """
        Extracts keywords from the text based on word frequency.

        :param text: The input text.
        :param top_n: The number of top keywords to return.
        :return: A list of keywords.
        """
        frequency = self.frequency_analysis(text)
        sorted_words = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]

    def frequency_analysis(self, text):
        """
        Performs frequency analysis of words in the text.

        :param text: The input text.
        :return: A dictionary with words as keys and their frequencies as values.
        """
        words = self.word_tokenize(text)
        frequency = {}
        for word in words:
            word_lower = word.lower()
            frequency[word_lower] = frequency.get(word_lower, 0) + 1
        return frequency

    def most_frequent_word(self, text):
        """
        Finds the most frequent word in the text.

        :param text: The input text.
        :return: The most frequent word.
        """
        frequency = self.frequency_analysis(text)
        if not frequency:
            return ""
        return max(frequency.items(), key=lambda item: item[1])[0]

    def least_frequent_word(self, text):
        """
        Finds the least frequent word in the text.

        :param text: The input text.
        :return: The least frequent word.
        """
        frequency = self.frequency_analysis(text)
        if not frequency:
            return ""
        return min(frequency.items(), key=lambda item: item[1])[0]

    def ngrams_char(self, text, n):
        """
        Generates character-based n-grams from the text.

        :param text: The input text.
        :param n: The length of each n-gram.
        :return: A list of n-grams.
        """
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def ngrams_word(self, text, n):
        """
        Generates word-based n-grams from the text.

        :param text: The input text.
        :param n: The number of words in each n-gram.
        :return: A list of n-grams.
        """
        words = self.word_tokenize(text)
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def shuffle_words(self, sentence):
        """
        Shuffles the words in a sentence.

        :param sentence: The input sentence.
        :return: The sentence with words shuffled.
        """
        import random
        words = self.word_tokenize(sentence)
        random.shuffle(words)
        return ' '.join(words)

    def reverse_word_order(self, sentence):
        """
        Reverses the order of words in a sentence.

        :param sentence: The input sentence.
        :return: The sentence with word order reversed.
        """
        words = self.word_tokenize(sentence)
        return ' '.join(reversed(words))

    def reverse_each_word(self, sentence):
        """
        Reverses each word in a sentence.

        :param sentence: The input sentence.
        :return: The sentence with each word reversed.
        """
        words = self.word_tokenize(sentence)
        reversed_words = [word[::-1] for word in words]
        return ' '.join(reversed_words)

    def newlines_to_br(self, text):
        """
        Converts newline characters to HTML <br> tags.

        :param text: The input text.
        :return: The text with newlines replaced by <br> tags.
        """
        return text.replace('\n', '<br>')

    def collapse_spaces(self, text):
        """
        Replaces multiple spaces with a single space.

        :param text: The input text.
        :return: The text with collapsed spaces.
        """
        result = ''
        prev_char = ''
        for char in text:
            if char != ' ' or prev_char != ' ':
                result += char
            prev_char = char
        return result

    def normalize_whitespace(self, text):
        """
        Normalizes whitespace in the text.

        :param text: The input text.
        :return: The text with normalized whitespace.
        """
        result = ''
        in_whitespace = False
        for char in text:
            if char in ' \t\n\r\f\v':
                if not in_whitespace:
                    result += ' '
                    in_whitespace = True
            else:
                result += char
                in_whitespace = False
        return result.strip()

    def trim_characters(self, text, chars):
        """
        Trims specific characters from the start and end of the text.

        :param text: The input text.
        :param chars: The characters to trim.
        :return: The trimmed text.
        """
        start = 0
        end = len(text) - 1
        while start <= end and text[start] in chars:
            start += 1
        while end >= start and text[end] in chars:
            end -= 1
        return text[start:end + 1]

    def remove_extra_newlines(self, text):
        """
        Removes extra newline characters from the text.

        :param text: The input text.
        :return: The text with extra newlines removed.
        """
        lines = text.split('\n')
        filtered_lines = [line for line in lines if line.strip()]
        return '\n'.join(filtered_lines)

    def is_palindrome(self, text):
        """
        Checks if the input text is a palindrome.

        :param text: The input text.
        :return: True if palindrome, False otherwise.
        """
        stripped = ''.join(char.lower() for char in text if char.isalnum())
        return stripped == stripped[::-1]

    def longest_palindromic_substring(self, text):
        """
        Finds the longest palindromic substring in the text.

        :param text: The input text.
        :return: The longest palindromic substring.
        """
        n = len(text)
        if n == 0:
            return ""

        start = 0
        max_length = 1

        for i in range(n):
            # Odd length palindromes
            low = i - 1
            high = i + 1
            while low >= 0 and high < n and text[low] == text[high]:
                if high - low + 1 > max_length:
                    start = low
                    max_length = high - low + 1
                low -= 1
                high += 1

            # Even length palindromes
            low = i
            high = i + 1
            while low >= 0 and high < n and text[low] == text[high]:
                if high - low + 1 > max_length:
                    start = low
                    max_length = high - low + 1
                low -= 1
                high += 1

        return text[start:start + max_length]


# Dummy test cases
if __name__ == "__main__":
    operations = StringOperations()

    # Soundex Algorithm
    print("Soundex of 'Example':", operations.soundex('Example'))

    # Metaphone Algorithm
    print("Metaphone of 'Example':", operations.metaphone('Example'))

    # Porter Stemming
    print("Stem of 'running':", operations.porter_stem('running'))

    # Pluralization
    print("Plural of 'leaf':", operations.pluralize('leaf'))

    # Singularization
    print("Singular of 'leaves':", operations.singularize('leaves'))

    # Converting to CamelCase
    print("CamelCase of 'hello world':", operations.to_camel_case('hello world'))

    # Converting to PascalCase
    print("PascalCase of 'hello world':", operations.to_pascal_case('hello world'))

    # Converting to snake_case
    print("snake_case of 'Hello World':", operations.to_snake_case('Hello World'))

    # Converting to kebab-case
    print("kebab-case of 'Hello World':", operations.to_kebab_case('Hello World'))

    # Sentence Splitting
    print("Sentences in text:", operations.sentence_split("Hello world! How are you? I'm fine."))

    # Word Tokenization
    print("Words in sentence:", operations.word_tokenize("Hello, world!"))

    # Counting Sentences
    print("Number of sentences:", operations.count_sentences("Hello world! How are you?"))

    # Counting Lines
    print("Number of lines:", operations.count_lines("Line one\nLine two\nLine three"))

    # Calculating Average Word Length
    print("Average word length:", operations.average_word_length("Hello world"))

    # Calculating Average Sentence Length
    print("Average sentence length:", operations.average_sentence_length("Hello world. How are you?"))

    # Identifying the Longest Word
    print("Longest word:", operations.longest_word("Hello amazing world"))

    # Identifying the Shortest Word
    print("Shortest word:", operations.shortest_word("An example"))

    # Sorting Words Alphabetically
    print("Sorted words:", operations.sort_words_alphabetically("banana apple cherry"))

    # Removing Stop Words
    print("Text without stop words:", operations.remove_stop_words("This is an example of stop words."))

    # Keyword Extraction
    print("Extracted keywords:", operations.extract_keywords("This is an example example example test test"))

    # Frequency Analysis of Words
    print("Word frequencies:", operations.frequency_analysis("This is a test. This test is easy."))

    # Finding Most Frequent Word
    print("Most frequent word:", operations.most_frequent_word("This is a test. This test is easy."))

    # Finding Least Frequent Word
    print("Least frequent word:", operations.least_frequent_word("This is a test. This test is easy."))

    # N-Gram Generation (Character-based)
    print("Character n-grams:", operations.ngrams_char("hello", 2))

    # N-Gram Generation (Word-based)
    print("Word n-grams:", operations.ngrams_word("This is a simple test", 2))

    # Shuffling Words in a Sentence
    print("Shuffled sentence:", operations.shuffle_words("Shuffle these words"))

    # Reversing Word Order
    print("Reversed word order:", operations.reverse_word_order("Reverse these words"))

    # Reversing Each Word
    print("Reversed each word:", operations.reverse_each_word("Reverse these words"))

    # Converting Newlines to <br> Tags
    print("Newlines to <br>:", operations.newlines_to_br("Line1\nLine2\nLine3"))

    # Collapsing Multiple Spaces
    print("Collapsed spaces:", operations.collapse_spaces("This   has    multiple spaces"))

    # Normalizing Whitespace
    print("Normalized whitespace:", operations.normalize_whitespace("This\t has  multiple\nwhitespaces"))

    # Trimming Specific Characters
    print("Trimmed text:", operations.trim_characters("...Hello World!!!", ".!"))

    # Removing Extra Newlines
    print("Removed extra newlines:", operations.remove_extra_newlines("Line1\n\n\nLine2\n\nLine3"))

    # Finding a Palindrome
    print("Is 'Madam' a palindrome?:", operations.is_palindrome("Madam"))

    # Extracting the Longest Palindromic Substring
    print("Longest palindromic substring:", operations.longest_palindromic_substring("babad"))