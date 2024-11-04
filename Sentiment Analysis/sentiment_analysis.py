from nltk.corpus import stopwords
from nltk import PorterStemmer
import threading
import queue
from load_data import *
import nltk
import pandas as pd
import os
import math 
from list_chunk_processor import distribute_task_across_cpus, reduce_to_single
import time
import string
from typing import Iterable, Any
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
import json

start = time.time()
def load_data(data_filename:str='data.csv', labels_filename:str='labels.csv'):
    """
    data_filename: Name of a csv files that contains the training and testing data
    The training data column name should be: trainData
    The testing data column name should be: testData

    labels_filename: Name of the csv files that contains the training and testing labels
    """
    paths = [data_filename, labels_filename]

    files_exist = all(os.path.isfile(get_path(path)) for path in paths)

    if files_exist:
        train_movie_data, test_movie_data, train_movie_labels, test_movie_labels = load_data_to_memory(data_filename=data_filename,
                                                                                                       labels_filename=labels_filename)
    else:
        train_movie_data, test_movie_data, train_movie_labels, test_movie_labels = download_and_load_data_to_memory()

    return train_movie_data, test_movie_data, train_movie_labels, test_movie_labels

def convert_text_to_vector(dataset:list[str], dataset_word_frequency:dict[str, dict]):
    """dataset: A list of texts"""
    if isinstance(dataset, str):
        dataset = [dataset]

    porter = PorterStemmer()
    vector = np.empty((0, 2))
    for text in dataset:
        # removing punctuation: Ideally there are some punctuations that are to be kept
        train_text_without_punctuation = text.translate(str.maketrans("", "", string.punctuation))

        # converting text to lowercase and splitting text.
        split_data = train_text_without_punctuation.lower().split(' ')

        # removing stopwords.
        split_text_without_stopwords = remove_stop_words(split_data)

        # remove repetitive words
        processed_words = set(split_text_without_stopwords)

        #print(processed_words)

        positive_freq_count = 0
        negative_freq_count = 0

        for word in processed_words:
            word = porter.stem(word)

            if word in dataset_word_frequency['positive']:
                positive_freq_count += dataset_word_frequency['positive'][word]
                #print(word, dataset_word_frequency['positive'][word])
                
            if word in dataset_word_frequency['negative']:
                negative_freq_count += dataset_word_frequency['negative'][word]
                #print(word, dataset_word_frequency['positive'][word])
        vector = np.vstack((vector, np.array([positive_freq_count, negative_freq_count])))

    return vector



def remove_stop_words(split_text:list[str]):
    #nltk.download('stopwords')

    stop_words = stopwords.words("english")

    split_text_wihtout_stopwords = list()

    for word in split_text:
        if word.lower() not in stop_words:
            split_text_wihtout_stopwords.append(word)
    return split_text_wihtout_stopwords


def process_data(movie_data:list[str]|str, labels:Iterable[int], load=False):
    """When using this function for multiprocessing set load to False"""
    if load:
        if os.path.isfile(get_path("frequency_dict.json")):
            return load_frequency_dict()

    if isinstance(movie_data, str):
        movie_data = [movie_data]
        labels = [labels]

    list_of_processed_words = list()

    frequency_dict = {"positive":dict(), "negative":dict()}

    porter = PorterStemmer()

    for train_text in movie_data:
        buffer_storage = list()

        # removing punctuation: Ideally there are some punctuations that are to be kept
        train_text_without_punctuation = train_text.translate(str.maketrans("", "", string.punctuation))

        # converting text to lowercase and splitting text.
        split_data = train_text_without_punctuation.lower().split(' ')

        # removing stopwords.
        split_text_without_stopwords = remove_stop_words(split_data)

        for word in split_text_without_stopwords:
            # performing stemming
            root_word  = porter.stem(word)

            buffer_storage.append(root_word)

        list_of_processed_words.append(buffer_storage)

    # getting frequency counts 
    for eachlist, label in zip(list_of_processed_words, labels):
        for word in eachlist:
            if label == 1: # postive statement
                if word in frequency_dict["positive"]:
                    frequency_dict['positive'][word] += 1 
                else:
                    frequency_dict["positive"][word] = 1
            else:
                if word in frequency_dict["negative"]:
                    frequency_dict['negative'][word] += 1 
                else:
                    frequency_dict["negative"][word] = 1

    if load: # change this later
        store_frequency_dict(frequency_dict)
    return frequency_dict

def process_data_across_cpus(movie_data:Iterable[Iterable[Any]], labels:Iterable[Iterable[int]], num_workers:int, load=True)->list[dict[str, dict[str, int]]]:
    """movie_data is a list that contains a list of your datasets e.g movie_data = [train_data, test_data]"""
    if load:
        if os.path.isfile(get_path("frequency_dict.json")):
            return [load_frequency_dict()]
        
    total_data_result:list[list[dict]] = distribute_task_across_cpus(process_data, movie_data, labels, num_workers)
    
    list_of_freq_dict_of_datasets = reduce_to_single(count_task, total_data_result, 2, num_workers)

    store_frequency_dict(list_of_freq_dict_of_datasets[0]) # change this later

    return list_of_freq_dict_of_datasets

def convert_to_vector_across_cpus(test_data:Iterable[Iterable[Any]], freq_dict:Iterable[dict], num_workers:int)->list[np.ndarray]:
    list_of_numpy_arrays = distribute_task_across_cpus(convert_text_to_vector, test_data, freq_dict, num_workers, constant_labels=True)
    vectors = reduce_to_single(stack_numpy_arrays, list_of_numpy_arrays, 2, num_workers)

    return vectors

def store_frequency_dict(frequency_dict):
    with open(get_path('frequency_dict.json'), 'w') as json_file:
        json.dump(frequency_dict, json_file)
    print("Frequency dict is stored")

def load_frequency_dict()->dict[str, dict[str, int]]:
    with open(get_path('frequency_dict.json'), 'r') as json_file:
        frequency_dict = json.load(json_file)
    print("Frequency dict is loaded")
    return frequency_dict

def count_task(*args):
    """Task for multiprocessing to count unique words in a dict using the Counter function
    args: a tuple contain an arbitrary length of dictionaries 
    """
    positive_frequency = Counter({})
    negative_frequency = Counter({})
    frequency_of_words = {'positive':dict(), 'negative': dict()}
    for frequency_dict in args:
        positive_frequency += Counter(frequency_dict['positive'])
        negative_frequency += Counter(frequency_dict['negative'])

    frequency_of_words['positive'] = dict(positive_frequency)
    frequency_of_words['negative'] = dict(negative_frequency)

    return frequency_of_words

def stack_numpy_arrays(*args:np.ndarray):
    """Multiprocessing task used for stacking the numpy arrays."""
    vector = np.empty((0, args[0].shape[1]))

    for array in args:
        vector = np.vstack((vector, array))
    return vector 


def convert_data_to_csv(train_movie_data, train_movie_labels):
    data = pd.DataFrame({
        'text': train_movie_data,
        'label': train_movie_labels
    })

    data.to_csv(get_path("sentiment_test_data.csv"), index=False)


if __name__ == "__main__":
    train_movie_data, test_movie_data, train_movie_labels, test_movie_labels = load_data()

    list_of_freq_dict_of_datasets = process_data_across_cpus([train_movie_data], [train_movie_labels], num_workers=10, load=True)

    vector = convert_to_vector_across_cpus([train_movie_data, test_movie_data], [list_of_freq_dict_of_datasets[0], list_of_freq_dict_of_datasets[0]], 10)
    train_labels = train_movie_labels.to_numpy()
    test_labels = test_movie_labels.to_numpy()

    cls = LogisticRegression(random_state=12, max_iter=500).fit(vector[0], train_labels)

    test_vector = vector[1]

    text_vector = convert_text_to_vector("The worst movie ever", list_of_freq_dict_of_datasets[0])

    prediction = cls.predict(text_vector)

    accuracy = cls.score(test_vector, test_labels)

    result_map = {0:'neg', 1:'pos'}

    print("Accuracy is", accuracy)

    print(f"prediction is {result_map[prediction.item()]}, True prediction is {result_map[test_movie_labels.iloc[0]]}")

    end = time.time()
    print(f"it took {end-start}s to finish")
