import pandas as pd
import numpy as np
import re
import arabicstopwords.arabicstopwords as ast
import string
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class MLTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            # If X is a Pandas Series, apply transformation to each element
            transformed_X = X.copy()
            transformed_X = transformed_X.apply(self.clean_text)
        else:
            # If X is text data, transform it directly
            transformed_X = self.clean_text(X)

        self.transformed_X = transformed_X
        return self.transformed_X

    def fit_transform(self, X, y=None):
        # This function combines fit and transform
        transformed_X = self.transform(X)
        return transformed_X

    def clean_text(self, text):
        # Remove emojis and symbols
        text = self.remove_emojis_and_symbols(text)

        # Remove URLs
        text = self.remove_url(text)

        # Unicode Normalization
        text = self.normalize_arabic(text)

        # Split hashtags
        text = self.split_hashtags(text)

        # Remove usernames
        text = self.remove_usernames(text)

        # Text normalization
        text = self.text_normalize(text)

        # Remove diactritics
        text = self.remove_diactritics(text)

        # Remove Punctations
        text = self.remove_punctuations(text)

        # Remove English
        text = self.remove_english(text)

        # Remove Special Characters
        text = self.remove_special_chars(text)

        # Remove digits
        text = self.remove_digits(text)

        # Remove elongation characters
        text = self.remove_elongation(text)

        # Remove tabs and newlines
        text = self.remove_extra_whitespaces(text)

        # Remove stop words
        text = self.remove_stop_words(text)

        return text

    def normalize_arabic(self, text, form='NFC'):
        """
        Normalize Arabic text using Unicode normalization.

        Args:
        - text (str): Input Arabic text.
        - form (str, optional): Unicode normalization form ('NFC' or 'NFD'). Default is 'NFC'.

        Returns:
        - str: Normalized Arabic text.
        """
        normalized_text = unicodedata.normalize(form, text)
        return normalized_text

    def remove_emojis_and_symbols(self, text):
        # Define the pattern to match emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U000024C2-\U0001F251"  # Other symbols to remove
            "\U000024C2-\U0001F251"
            "\U000fe300-\U000fe3FF"
            "\U000e0060-\U000e0069"
            "\U000fe000-\U000fe999"
            "\U000fe300-\U000fe3FF"
            "\U000feb10-\U000feb99"
            "\U000fe10e-\U000fe1de"
            "\U000feba0-\U000fec99"
            "\U000e006e-\U000e00ff"
            "\U0009fc0d-\U000febff"
            "\U0001F300-\U0001F6FF"
            "\u2060-\u2069"
            "\u200a-\u200f"
            "\u061c"
            "\u23ea"
            "]+", flags=re.UNICODE
        )

        return emoji_pattern.sub(r'', text)

    def split_hashtags(self, text):
        pattern = r'#\w+'
        hashtags = re.findall(pattern, text)
        for hashtag in hashtags:
            # Remove the '#' symbol and split the remaining text
            hashtag_text = hashtag[1:].replace("_", " ")  # Remove '#' and replace '_' with space
            # Replace the hashtag with the split version in the text
            text = text.replace(hashtag, hashtag_text)
        return text

    def remove_usernames(self, text):
        pattern = r'\@\S+'
        return re.sub(pattern, r'', text)

    def remove_elongation(self, text):
        pattern = re.compile(r'ـ+')
        # Replace elongation characters with an empty string
        return re.sub(pattern, '', text)

    def text_normalize(self, text):
        # Remove redundant letters
        text = re.sub(r'(.)\1+', r"\1", text)

        # Text normalization
        text = re.sub("[إأٱآااً]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ى", "ٸ", text)
        text = re.sub("[ؤٶ]", "ء", text)
        text = re.sub("[ۄۆ]", "و", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("۽", "ء", text)
        text = re.sub("ڈ", "د", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("[ڱګگݣکڪڳڭ]", "ك", text)
        text = re.sub("ڤ", "ف", text)
        text = re.sub("ڨ", "ف", text)
        text = re.sub("چ", "ج", text)
        text = re.sub("ژ", "ز", text)
        text = re.sub("ڒ", "ز", text)
        text = re.sub("ٺ", "ت", text)
        text = re.sub("پ", "ب", text)

        return text

    def remove_stop_words(self, text):
        stop_words = ast.stopwords_list()
        return ' '.join(word for word in text.split() if word not in stop_words)

    def remove_digits(self, text):
        text = re.sub(r'[0-9]+', '', text)  # Remove English digits
        text = re.sub(r'[٠١۲٣٤٥٦٧٨٩]+', '', text)  # Remove Arabic digits
        return text

    def remove_url(self, text):
        pattern = r'ht*ps?://(?:www\.)?\S+'
        return re.sub(pattern, ' ', text)

    def remove_diactritics(self, text):
        pattern = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ   |  # Tatwil/Kashida

                             """, re.VERBOSE)
        return re.sub(pattern, '', text)

    def remove_extra_whitespaces(self, text):
        return re.sub(r"\s+", ' ', text)

    def remove_punctuations(self, text):
        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ‼'''
        english_punctuations = string.punctuation
        punctuations_list = arabic_punctuations + english_punctuations
        translator = str.maketrans('', '', punctuations_list)
        return text.translate(translator)

    def remove_special_chars(self, text):
        pattern = re.compile(r'[⇸Áäнפ©дےۓǑ⏱Ųлḁ℅ιş¤Ỳतदᾰ₩स₩⑷⌣ыИ②ღŠπƸ£⁉àéáê•тô¿ç™в§йцस«↑עह€и②√Ӝक]+')
        return re.sub(pattern, '', text)

    def remove_english(self, text):
        english_pattern = re.compile(r'[a-zA-Z]+')
        return re.sub(english_pattern, '', text)


class DLTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            # If X is a Pandas Series, apply transformation to each element
            transformed_X = X.copy()
            transformed_X = transformed_X.apply(self.clean_text)
        else:
            # If X is text data, transform it directly
            transformed_X = self.clean_text(X)

        self.transformed_X = transformed_X
        return self.transformed_X

    def fit_transform(self, X, y=None):
        # This function combines fit and transform
        transformed_X = self.transform(X)
        return transformed_X

    def clean_text(self, text):
        # Remove emojis and symbols
        text = self.replace_emojis_and_symbols(text)

        # Remove URLs
        text = self.replace_url(text)

        # Unicode Normalization
        text = self.normalize_arabic(text)

        # Split hashtags
        text = self.split_hashtags(text)

        # Remove usernames
        text = self.replace_usernames(text)

        # Remove elongation characters
        text = self.remove_elongation(text)

        # Remove Special Characters
        text = self.remove_special_chars(text)

        # Text normalization
        text = self.text_normalize(text)

        # Remove diactritics
        text = self.remove_diactritics(text)

        # Remove digits
        text = self.replace_digits(text)

        # Remove tabs and newlines
        text = self.remove_extra_whitespaces(text)

        return text

    def normalize_arabic(self, text, form='NFC'):
        """
        Normalize Arabic text using Unicode normalization.

        Args:
        - text (str): Input Arabic text.
        - form (str, optional): Unicode normalization form ('NFC' or 'NFD'). Default is 'NFC'.

        Returns:
        - str: Normalized Arabic text.
        """
        normalized_text = unicodedata.normalize(form, text)
        return normalized_text

    def replace_emojis_and_symbols(self, text):
        # Define the pattern to match emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U000024C2-\U0001F251"  # Other symbols to remove
            "\U000024C2-\U0001F251"
            "\U000fe300-\U000fe3FF"
            "\U000e0060-\U000e0069"
            "\U000fe000-\U000fe999"
            "\U000fe300-\U000fe3FF"
            "\U000feb10-\U000feb99"
            "\U000fe10e-\U000fe1de"
            "\U000feba0-\U000fec99"
            "\U000e006e-\U000e00ff"
            "\U0009fc0d-\U000febff"
            "\u2060-\u2069"
            "\u200a-\u200f"
            "\u061c"
            "\u23ea"
            "]+", flags=re.UNICODE
        )

        return emoji_pattern.sub(r'EMOJI', text)

    def split_hashtags(self, text):
        pattern = r'#\w+'
        hashtags = re.findall(pattern, text)
        for hashtag in hashtags:
            # Remove the '#' symbol and split the remaining text
            hashtag_text = hashtag[1:].replace("_", " ")  # Remove '#' and replace '_' with space
            # Replace the hashtag with the split version in the text
            text = text.replace(hashtag, hashtag_text)
        return text

    def replace_usernames(self, text):
        pattern = r'\@\S+'
        return re.sub(pattern, r'@USER', text)

    def remove_elongation(self, text):
        pattern = re.compile(r'ـ+')
        # Replace elongation characters with an empty string
        return re.sub(pattern, '', text)

    def text_normalize(self, text):
        # Remove redundant letters
        text = re.sub(r'(.)\1+', r"\1", text)

        # Text normalization
        text = re.sub("[إأٱآااً]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ى", "ٸ", text)
        text = re.sub("[ؤٶ]", "ء", text)
        text = re.sub("[ۄۆ]", "و", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("۽", "ء", text)
        text = re.sub("ڈ", "د", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("[ڱګگݣکڪڳڭ]", "ك", text)
        text = re.sub("ڤ", "ف", text)
        text = re.sub("ڨ", "ف", text)
        text = re.sub("چ", "ج", text)
        text = re.sub("ژ", "ز", text)
        text = re.sub("ڒ", "ز", text)
        text = re.sub("ٺ", "ت", text)
        text = re.sub("پ", "ب", text)

        return text

    def replace_digits(self, text):
        text = re.sub(r'[0-9]+', 'NUM', text)  # Remove English digits
        text = re.sub(r'[٠١۲٣٤٥٦٧٨٩]+', 'NUM', text)  # Remove Arabic digits
        return text

    def replace_url(self, text):
        pattern = r'ht*ps?://(?:www\.)?\S+'
        return re.sub(pattern, 'URL', text)

    def remove_diactritics(self, text):
        pattern = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ   |  # Tatwil/Kashida

                             """, re.VERBOSE)
        return re.sub(pattern, '', text)

    def remove_extra_whitespaces(self, text):
        return re.sub(r"\s+", ' ', text)

    def remove_special_chars(self, text):
        pattern = re.compile(r'[‹⇸Ùप⑵ब»Áäнפ©дےۓǑ⏱Ųлḁ℅ιş¤Ỳतदᾰ₩स₩⑷⌣ыИ②ღŠπƸ£⁉àéáê•тô¿ç™в§йцस«↑עह€и②√Ӝक]+')
        return re.sub(pattern, '', text)
