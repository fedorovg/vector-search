from typing import *
from prepocessor import Preprocessor
from collections import Counter
import numpy as np
import os


class SearchEngine:
    
    def __init__(self, documents_path: str, k=1):
        self.documents_path = documents_path
        # throughout this file tf refers to term frequency (number of occurrences of a word)
        self.doc_count = 0
        # Tuning parameter k: Should be decreased for datasets with large documents and increased for the small ones
        self.k = k
        self.vector_dim = 0  # Number of unique terms
        self.mapping = {}  # Mapping of a word to index in a vector
        self.tfs = []  # vectors for each document
        self.idf = None  # Inverse document frequencies
        self.average_doc_length = 0
    
     def generate_vectors(self):
        """
        Sets model up by generating vectors from documents and precomputing additional data. (e.g. idf)
        :return:
        """
        counts, global_counts = self.get_term_counts()
        # After counting all terms in our documents, we can create a mapping and decide on the vector dimentions
        self.vector_dim = len(global_counts)
        self.mapping = self.create_mapping(global_counts)
        self.tfs = self.counts_to_vectors(counts)
        self.doc_count = len(self.tfs)
        self.idf = self.calculate_idf(self.tfs)
        self.average_doc_length = np.sum(sum([v for _, v in self.tfs])) / self.doc_count

    def get_term_counts(self) -> Tuple[Dict[str, Counter], Counter]:
        """
        Counts occurrences of each word in the whole document set and per document.
        These counts will be used to calculate values of the vectors
        :return: Term frequencies for each document and global tfs.
        """
        global_tf = Counter({})
        doc_tfs = {}
        for document in os.listdir(self.documents_path):
            with open(f"{self.documents_path}/{document}", "r") as f:
                words = [
                    word for line in f.readlines()
                    for word in line.split()
                ]
                doc_tf = Counter(words)
                doc_tfs[document] = doc_tf
                global_tf += doc_tf
        
        return doc_tfs, global_tf
    
      def counter_to_vector(self, counter: Counter) -> np.ndarray:
        """
        Creates a vector from a Counter according to the self.mapping
        :param counter:
        :return: A tf vector
        """
        res = np.zeros(self.vector_dim)
        for term, count in counter.items():
            res[self.mapping[term]] = count
        return res
    
    def counts_to_vectors(self, counts: Dict[str, Counter]) -> List[Tuple[str, np.ndarray]]:
        """
        This method transforms word counts into tf vectors
        :param counts:
        :return:
        """
        return [
            (k, self.counter_to_vector(v)) for k, v in counts.items()
        ]
  
    def create_mapping(self, global_tf: Counter) -> Dict[str, int]:
        """
        Creates a mapping from each word to it's index inside a vectors.
        :return:
        """
        words = global_tf.keys()
        return dict(
            zip(words, range(self.vector_dim))
        )
    
    @staticmethod
    def calculate_idf(tfs: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """
        Computes inverse document frequency for all terms. (Number of documents that contain a term)
        :return: Vector of idfs
        """
        df = sum([vec.clip(max=1) for (_, vec) in tfs])
        return np.log(len(tfs) / df)  # df can not contain zeros, since any word is present in at least one document

   

if __name__ == "__main__":
    engine = SearchEngine("data/articles/processed")
