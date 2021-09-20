from typing import *

from collections import Counter
import os


class SearchEngine:
    
    def __init__(self, documents_path: str):
        self.documents_path = documents_path
        self.counts = Counter({})
        self.mapping = {}

    def count_all_words(self):
        """
        Counts occurrences of each word in the whole document set.
        These counts will be used to calculate values of the vectors
        :return:
        """
        for document in os.listdir(self.documents_path):
            with open(f"{self.documents_path}/{document}", "r") as f:
                words = [
                    word for line in f.readlines()
                    for word in line.split()
                ]
                self.counts += Counter(words)
                
    def map_words_to_index(self):
        """
        Creates a mapping from each word to it's index inside of a vector.
        :return:
        """
        words = self.counts.keys()
        self.mapping = dict(
            zip(
                words, range(len(words))
            )
        )
    
        
    


if __name__ == "__main__":
    engine = SearchEngine("data/articles/processed")
    engine.count_all_words()
    engine.map_words_to_index()
    print(engine.mapping)
