from collections import Counter
import pandas as pd
import numpy as np
import nltk
import string
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import Iterable

def is_data_downloaded():
    """Checks if the stopwords and wordnet (for lemmatization) are downloaded"""

    for resource in ["wordnet", "stopwords"]:

        try:
            nltk.data.find(f'corpora/{resource}.zip')
        except LookupError:
            nltk.download(resource)

def remove_stop_words(split_text:list[str]):
    
    stop_words = stopwords.words("english")

    split_text_wihtout_stopwords = list()

    for word in split_text:
        if word.lower() not in stop_words:
            split_text_wihtout_stopwords.append(word)
    return split_text_wihtout_stopwords

def preprocess_text(sentence:str):
    lemmatizer = WordNetLemmatizer()

    split_sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    split_sentence = split_sentence.lower().split()
    split_sentence = remove_stop_words(split_sentence)

    split_sentence = [lemmatizer.lemmatize(word) for word in split_sentence]

    return split_sentence


def get_unique_words(dataset:Iterable[str]):
    if isinstance(dataset, str):
        dataset = [dataset]
        
    unique_words = set()
    for text in dataset:
        split_text = preprocess_text(text)
        unique_words.update(split_text)
    return unique_words

 
def bow(data:Iterable[str]):

    # download neccessary data if not already downloaded
    is_data_downloaded()

    # map words in vocab to indices
    vocab = get_unique_words(data)
    vocab = sorted(vocab)
    vocab_index = {word:i for i, word in enumerate(vocab)}

    # initialize numpy array (preallocation)
    col = len(vocab)
    row = len(data)
    word_count = np.zeros(shape=(row, col), dtype=int)

    for idx, sentence in enumerate(data):

        split_sentence = preprocess_text(sentence)
        counted_words = Counter(split_sentence)

        for word, count in counted_words.items():
            word_count[idx, vocab_index[word]] = count


    bag_of_words = pd.DataFrame(data=word_count, columns=vocab)

    return bag_of_words


if __name__ == "__main__":
    data = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
    ]
    bag_of_words = bow(data)
    print(bag_of_words)