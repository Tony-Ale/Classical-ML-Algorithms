from sklearn.datasets import load_files
from nltk.corpus import stopwords
import nltk
import pandas as pd
import os

def get_path(file_path:str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data", file_path)
    return path

def download_and_load_data_to_memory():
    train_movie_path = get_path(file_path="aclimdb_v1/aclimdb/train")
    test_movie_path = get_path(file_path="aclimdb_v1/aclimdb/test")

    train_movie_data_dict = load_files(container_path=train_movie_path, categories=['neg', 'pos'])

    test_movie_data_dict = load_files(container_path=test_movie_path, categories=['neg', 'pos'])

    train_movie_data_byte = train_movie_data_dict.data
    test_movie_data_byte = test_movie_data_dict.data

    train_movie_labels = train_movie_data_dict.target
    test_movie_labels = test_movie_data_dict.target


    train_movie_data = [str(data, encoding="utf-8") for data in train_movie_data_byte]
    test_movie_data = [str(data, encoding="utf-8") for data in test_movie_data_byte]

    print("Files downloaded")

    data = cache_files_and_load_data_to_memory(train_movie_data, 
                                                test_movie_data, 
                                                train_movie_labels, 
                                                test_movie_labels)

    return data

def cache_files_and_load_data_to_memory(train_movie_data, test_movie_data, train_movie_labels, test_movie_labels):

    # loadind data to a dataframe
    df_data = pd.DataFrame({'trainData':train_movie_data, 'testData':test_movie_data})

    # Loading labels to csv because they are numpy arrays 
    df_labels = pd.DataFrame({'trainLabels':train_movie_labels, 'testLabels':test_movie_labels})

    df_labels.to_csv(get_path("labels.csv"), index=False)
    df_data.to_csv(get_path("data.csv"), index=False)

    print("Files cached and loaded to memory")

    return df_data['trainData'], df_data['testData'], df_labels['trainLabels'], df_labels['testLabels']

def load_data_to_memory(data_filename:str, labels_filename:str):
    # load data
    df_data = pd.read_csv(get_path(data_filename))
    train_movie_data = df_data[df_data.columns[0]]
    test_movie_data = df_data[df_data.columns[1]]
    
    # load labels 
    df_labels = pd.read_csv(get_path(labels_filename))
    train_movie_labels = df_labels[df_labels.columns[0]]
    test_movie_labels = df_labels[df_labels.columns[1]]

    print("Files loaded successfully")
    return train_movie_data, test_movie_data, train_movie_labels, test_movie_labels



    

    

    

