import math
import threading
import queue 
def put_task_in_thread(task:callable, train_data:list, test_data:list, est_job_per_worker_train:list, est_job_per_worker_test:list, num_workers:int):
    # Error checking not actually needed but i just implemented it to ensure my algorithms always work well
    if est_job_per_worker_train[-1] != len(train_data) or est_job_per_worker_test[-1] != len(test_data):
        ValueError("The total number of jobs the workers are to do is not equal to the job available")
    elif(num_workers != len(est_job_per_worker_train) or num_workers != len(est_job_per_worker_test)):
        ValueError("Number of jobs is more than the requested number of workers")

    def worker_task(data_queue, data):
        with semaphore:
            task_output = task(data)
            data_queue.put(task_output)

    max_threads = 10  # Set your desired maximum number of threads
    semaphore = threading.Semaphore(max_threads)

    threads = []
    train_queue= queue.Queue()
    test_queue= queue.Queue()

    # for train dataset
    start_index = 0
    for stop_index in est_job_per_worker_train: # the length of this list is the number of workers asked for.
        thread = threading.Thread(target=worker_task, args=(train_queue, train_data[start_index:stop_index]))
        threads.append(thread)
        thread.start()
        start_index = stop_index

    # for test dataset
    start_index = 0
    for stop_index in est_job_per_worker_test: # the length of this list is the number of workers asked for.
        thread = threading.Thread(target=worker_task, args=(test_queue, test_data[start_index:stop_index]))
        threads.append(thread)
        thread.start()
        start_index = stop_index

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect and print results
    train_vocab = []
    test_vocab = []
    while not train_queue.empty():
        train_vocab.append(train_queue.get())


    while not test_queue.empty():
        test_vocab.append(test_queue.get())

    print("All tasks completed.")
    print(train_vocab)
    return train_vocab, test_vocab

def multi_threading(num_workers=5):
    def decorator(task:callable):
        def split_data(train_data, test_data):
            # place condition for when length of data is less than the num_workers, although this will never happen
            train_len = len(train_data)
            test_len = len(test_data)

            est_job_per_worker_train = math.floor(train_len/num_workers)
            est_job_per_worker_test = math.floor(test_len/num_workers)

            # Assuming est_job_per_worker_train = 2, the list generated will be [2, 4, 6, 8, 10].
            # Each element in the list represents the ending index (element[n]-1) of the data assigned to each worker.
            # This means that for 5 workers, the first worker is responsible for data up to index 1, the second worker up to index 3 (starting from index 2), and so on.

            if train_len % num_workers != 0:
                extra_job_for_last_worker = train_len - (est_job_per_worker_train * num_workers)
                est_job_per_worker_train = [est_job_per_worker_train*i for i in range(1, num_workers+1)]
                est_job_per_worker_train[-1] += extra_job_for_last_worker
            else:
                est_job_per_worker_train = [est_job_per_worker_train*i for i in range(1, num_workers+1)]

            if test_len % num_workers != 0:
                extra_job_for_last_worker = test_len - (est_job_per_worker_test * num_workers)
                est_job_per_worker_test = [est_job_per_worker_test*i for i in range(1, num_workers+1)]
                est_job_per_worker_test[-1] += extra_job_for_last_worker
            else:
                est_job_per_worker_test = [est_job_per_worker_test*i for i in range(1, num_workers+1)]

            return put_task_in_thread(task, train_data, test_data, est_job_per_worker_train, est_job_per_worker_test, num_workers)

        return split_data 
    return decorator

@multi_threading(5)
def task(data:list):
    data.append("i have been processed")
    return data

if __name__ == "__main__":
    data1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    data2 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 ,3 ]
    train, test = task(data1, data2)
    print(test)