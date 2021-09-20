from typing import *

import pandas as pd
import nltk


def split_csv_into_articles_and_process(csv_path: str,
                                        stopwords_path="./data/stopwords.txt",
                                        max_articles_to_extract: int = 100) -> None:
    """
    This function runs whole preprocessing pipeline on each row in csv
    and stores both raw and processed versions to the filesystem.
    :param csv_path: path to csv file with the dataset
    :param stopwords_path: path to a file with additional stopwords
    :param max_articles_to_extract: limits number of documents, that will be extracted
    :return:
    """
    df = pd.read_csv(csv_path)
    p = Preprocessor(additional_stopwords_path=stopwords_path)
    for i, row in df.iterrows():
        if i >= max_articles_to_extract:
            break
        
        article_name = row.title.replace("/", "")
        raw_path = f"./data/articles/raw/{article_name}"
        processed_path = f"./data/articles/processed/{article_name}"
        
        with open(raw_path, "w+") as f:
            f.write(row.content)
        
        with open(processed_path, "w+") as f:
            processed = " ".join(p.process(row.content))
            f.write(processed)


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
    
    def process(self, raw_text: str) -> List[str]:
        """
        Runs the whole preprocessing pipeline.
        :param raw_text:
        :return: List of tokens
        """
        return list(
            self.stem(self.tokenize(raw_text))
        )


if __name__ == "__main__":
    split_csv_into_articles_and_process("./data/articles1.csv", max_articles_to_extract=1)
