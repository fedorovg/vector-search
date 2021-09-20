from typing import *
import nltk


class Preprocessor:
    """
        Preprocessor is a class that encapsulates all logic and state required to turn documents into a list of tokens.
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
                
        self.stemmer = nltk.stem.PorterStemmer()

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
    
    def stem(self, tokens: Iterable[str]) -> Iterable[str]:
        """
        This method removes all suffixes from tokens, leaving only a stem.
        Resulting stems are not always valid words,
        but it is ok, since all of the documents will be processed the same way.
        :param tokens:
        :return: Iterable of token stems
        """
        return map(self.stemmer.stem, tokens)


if __name__ == "__main__":
    p = Preprocessor()
    example = "First sentences. Second And one. Running"
    print(list(p.stem(p.tokenize(example))))
