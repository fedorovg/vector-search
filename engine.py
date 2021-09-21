from typing import *
from prepocessor import Preprocessor
from collections import Counter
import numpy as np
import os


class SearchEngine:
    """
    SearchEngine is a class that represents a vector space model.
    It can be used to search for documents similar to a request srting.
    """
    
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
    
    def find_matches(self, query: List[str], top=10):
        """
        This method find top n documents that are most similar to the query.
        :param query:
        :param top:
        :return: list of document names
        """
        if not query:
            return []
        try:
            q_tf = self.vectorize_query(query)
        except Exception:
            return []
        best_matches = sorted(self.tfs, key=lambda doc: self.calculate_similarity(q_tf, doc[1]), reverse=True)[:top]
        return [name for name, _ in best_matches]
    
    def calculate_similarity(self, q_tf: np.ndarray, d_tf: np.ndarray):
        """
        Calculates similarity between a query vector and a document and expresses it as a number.
        :param q_tf: query vector
        :param d_tf: document vector
        :return: similarity number
        """
        freq_norm = np.sum(d_tf) / (self.average_doc_length * self.k + d_tf)
        return np.sum(q_tf * (d_tf / freq_norm) * self.idf)
    
    def vectorize_query(self, query_terms: List[str]) -> np.ndarray:
        """
        Takes a query and transforms it into a tf vector.
        :param query_terms:
        :return:
        """
        try:
            return self.counter_to_vector(Counter(query_terms))
        except KeyError:
            print("Query is wildly different from anything we have")
            raise Exception("Cant find")
    
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
    engine.generate_vectors()
    p = Preprocessor()
    q = p.process("trump")
    print(engine.find_matches(q))
