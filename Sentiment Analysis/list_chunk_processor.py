import math
import threading
import queue 
from multiprocessing import Process, Queue, Pool, set_start_method
from typing import Any, Iterable
import os

def distribute_task_across_cpus(task:callable, datasets:list[list[Any]], labels:Iterable[Iterable[int]], num_workers:int, constant_labels=False):
    """
    Takes in a list of data you want to process and process them using a function given to it 
    datasets: A list containing lists of data list[list[Any]]
    args: allows you to add other data set for processing
    task: the function that process the data
    """
    print("Processing across cpus...")
    cpu_count = os.cpu_count()
    num_workers = min(num_workers, cpu_count)
    total_data_result = list()
    
    with Pool(processes=num_workers) as pool:
        for data, label in zip(datasets, labels):
            if constant_labels:
                arg = ((datapoint, label) for datapoint in data)
            else:
                arg = zip(data, label)

            chunk_size = max(1, int(len(data)/num_workers))
            data_vocab = pool.starmap(task, arg, chunksize=chunk_size)
            total_data_result.append(data_vocab)

    print("All tasks completed.\n")
    return total_data_result

def divisions_until_one(n:int, divisor:int):
    """logarithims can also be used
    How many times a number will be divided until it gets to one, if a fraction is gotten, the nearest integer above the fraction is used
    """
    count = 0
    while n > 1:
        n = math.ceil(n / divisor)
        count += 1
    return count

def reduce_to_single(task:callable, datasets:list[list[Any]], chunk_to_sum:int, num_workers:int,):
    """Reduces the elements of a list to a single element, the list being reduced is a sublist of dataset"""
    print("processing across cpus...")
    cpu_count = os.cpu_count()
    num_workers = min(num_workers, cpu_count)

    with Pool(processes=num_workers) as pool:
        
        total_data_result = list()
        for result_per_dataset in datasets:

            count = divisions_until_one(len(result_per_dataset), chunk_to_sum)

            for _ in range(count):
                chunked_result = [result_per_dataset[i:i+chunk_to_sum] for i in range(0, len(result_per_dataset), chunk_to_sum) ]
                output = pool.starmap(task, chunked_result)
                result_per_dataset = output


            # To confirm result 
            if len(output) == 1:
                print("List has been reduced to a single element")
            else:
                raise Exception("The length of the List is not one")

            total_data_result.append(output[0])
    
    print("All tasks completed.\n")
    
    return total_data_result

