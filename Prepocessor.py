from typing import *
import nltk


class Preprocessor:
    """
        Preprocessor is a class that encapsulates all logic and state required to
    """
    
    def __init__(self, language="english", additional_stopwords_path=""):
        try:
            # Download nltk packages if not present
            nltk.data.find("tokenizers/punkt")
            nltk.data.find('corpora/stopwords')
        except:
            nltk.download("punkt")
            nltk.download("stopwords")
        
        self.stop_words = nltk.corpus.stopwords.words(language)
        if additional_stopwords_path:
            with open(additional_stopwords_path, "r") as f:
                self.stop_words = self.stop_words + [w.strip("\n") for w in f.readlines()]
    
    def tokenize(self, text: str) -> Iterable[str]:
        """
        This method converts a string of raw text into a list of lowercase tokens without stopwords and punctuation
        :param text: A raw string of text.
        :return: Iterable of tokens
        """
        return filter(lambda x: x not in self.stop_words, [
            tok.lower()
            for tok in nltk.word_tokenize(text)
            if tok.isalnum()
        ])


if __name__ == "__main__":
    p = Preprocessor()
    example = "First sentences. Second And one. Running"
    print(list(p.tokenize(example)))
