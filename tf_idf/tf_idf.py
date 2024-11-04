from typing import Iterable
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import nltk
import string
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, lil_matrix, issparse
from scipy.sparse.linalg import norm

class tf_idf:
    def __init__(self) -> None:
        self.idf_:np.ndarray
        self.tf_:csr_matrix
        self.col_to_index: dict

        self.is_data_downloaded()

    def is_data_downloaded(self):
        """Checks if the stopwords and wordnet (for lemmatization) are downloaded"""

        for resource in ["wordnet", "stopwords"]:
            try:
                nltk.data.find(f'corpora/{resource}.zip')
            except LookupError:
                nltk.download(resource)

    def remove_stop_words(self, split_text:list[str]):
        
        stop_words = stopwords.words("english")

        split_text_wihtout_stopwords = list()

        for word in split_text:
            if word.lower() not in stop_words:
                split_text_wihtout_stopwords.append(word)
        return split_text_wihtout_stopwords

    def tokenize_text(self, sentence:str):
        lemmatizer = WordNetLemmatizer()

        split_sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        split_sentence = split_sentence.lower().split()
        split_sentence = self.remove_stop_words(split_sentence)

        split_sentence = [lemmatizer.lemmatize(word) for word in split_sentence]

        return split_sentence

    def get_feature_names_out(self, documents:Iterable[str]):
        if isinstance(documents, str):
            documents = [documents]
            
        unique_words = set()
        for text in documents:
            split_text = self.tokenize_text(text)
            unique_words.update(split_text)
        return unique_words

    def count_documents_with_word(self, documents:Iterable[str]):
        number_of_doc_containing_term = defaultdict(int)
        
        for doc in documents:
            processed_doc = self.tokenize_text(doc)

            unique_words = set(processed_doc)

            for word in unique_words:
                number_of_doc_containing_term[word] += 1

        # convert to numpy array
        sorted_dict  = dict(sorted(number_of_doc_containing_term.items()))
        docs_per_term_array = np.array(list(sorted_dict.values()))

        return docs_per_term_array


    def _calculate_idf(self, documents:Iterable[str]):
        docs_per_term_array = self.count_documents_with_word(documents)

        total_number_of_doc = len(documents)

        docs_per_term_array = np.log((total_number_of_doc + 1)/(docs_per_term_array + 1)) + 1

        self.idf_ = docs_per_term_array

    def _calculate_tf(self, documents:Iterable[str]):

        # creating sparse array list of list
        row = len(documents)
        col = len(self.col_to_index)
        self.tf_ = lil_matrix((row, col), dtype=float)

        for idx, document in enumerate(documents):
            processed_doc = self.tokenize_text(document)

            terms_count_in_doc = Counter(processed_doc) # A counter object has similar behaviour to a default dict, instead of raising a key error, it initializes that key with 0
            
            number_of_terms_in_doc = len(processed_doc) # getting the total number of words

            for key in terms_count_in_doc.keys():
                if key in self.col_to_index:
                    # calculating term frequency
                    self.tf_[idx, self.col_to_index[key]] = terms_count_in_doc[key]/number_of_terms_in_doc

        # converting to a sparse matrix, to make storage and computation efficient
        self.tf_ = self.tf_.tocsr()#csr_matrix(self.tf_)

    def fit(self, documents:Iterable[str]):
        feature_names = self.get_feature_names_out(documents) # gets all unique words in present in the list of documents 
        feature_names = sorted(feature_names)

        self.col_to_index = {col_name: idx for idx, col_name in enumerate(feature_names)}

        self._calculate_idf(documents)

        return self
    
    def transform(self, documents:Iterable):
        # calculate term frequency 
        self._calculate_tf(documents)

        scores = self.tf_.multiply(self.idf_)
        norm_array = np.sqrt(scores.multiply(scores).sum(axis=1))

        # To prevent zero division error
        norm_array[norm_array==0] = 1
        self.tf_idf_scores = scores/norm_array
        return self.tf_idf_scores
    
    def fit_transform(self, documents:Iterable):
        obj = self.fit(documents)
        tf_idf_scores = obj.transform(documents)
        return tf_idf_scores
    
    def visualize_data(self, array:csr_matrix|np.ndarray):
        """Converts data to a pandas dataframe for better visualization and interpretability"""
        index = None
        if array.ndim == 1:
            array = np.reshape(array, (1, -1))
        else:
            index = [f"doc {idx}" for idx in range(1, array.shape[0]+1)]

        if issparse(array): # if it is a sparse matrix
            array = array.toarray()

        df = pd.DataFrame(array, columns=list(self.col_to_index.keys()), index=index)

        return df
            

if __name__ == "__main__":
    documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
    ]
    tf_idfs = tf_idf()
    obj = tf_idfs.fit(documents)
    scores = obj.transform(documents)
    print(scores.toarray())

    df = tf_idfs.visualize_data(scores)

    print(df)
