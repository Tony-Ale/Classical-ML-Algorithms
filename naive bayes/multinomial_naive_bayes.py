import numpy as np
from typing import Iterable

class multinomialNB:

    def label_map(self, y:np.ndarray):
        label_map  = dict()
        classes = np.unique(y) # the unique() function automatically sorts the unique values in ascending order
        for idx, eachclass in enumerate(classes):
            label_map[f'class_{idx}'] = eachclass
        return label_map

    def process_data(self, X:np.ndarray|Iterable[str] , y:np.ndarray):
        self.label_map = self.label_map(y)

        initial_class_count = {class_id: 1 for class_id in self.label_map}
        
        frequency_dict = dict()

        for text, label in zip(X, y):
            # to eliminate other occurence of a word in the text we have to convert it to a set
            splitted_text = set(text.lower().split(' '))
            for word in splitted_text:
                if word not in frequency_dict:
                    frequency_dict[word] = initial_class_count.copy() #{'spam':1, 'notspam':1} # give all words to have this initial value
                for class_id in self.label_map: 
                    if word in frequency_dict:
                        if label == self.label_map[class_id]:
                            frequency_dict[word][class_id] += 1

        return frequency_dict

    def fit(self, X:np.ndarray|Iterable[str] , y:np.ndarray):
        frequency_dict = self.process_data(X, y)
        initial_prob = {class_id: None for class_id in self.label_map}
        
        num_samples = y.size

        self.likelihoods:dict[str, dict] = dict()
        self.priors = initial_prob.copy()

        for class_id in self.label_map:
            specific_class = y[y==self.label_map[class_id]]
            len_specific_class = specific_class.size

            pr_specific_class = len_specific_class/num_samples # prior
            self.priors[class_id] = pr_specific_class

            for word in frequency_dict:
                if word not in self.likelihoods:
                    self.likelihoods[word] = initial_prob.copy()
                if word in self.likelihoods:
                    self.likelihoods[word][class_id] = frequency_dict[word][class_id]/len_specific_class

    def predict(self, X:str |np.ndarray | Iterable[str]):
        if isinstance(X, str):
            X = [X]

        predictions = np.empty((0, len(self.label_map)))

        #print(self.likelihoods)

        for text in X:
            array_of_probabilities = np.empty((0, len(self.label_map)))
            # Adding priors 
            array_of_probabilities = np.vstack((array_of_probabilities, np.array(list(self.priors.values()))))

            splitted_text = set(text.lower().split(' '))
            for word in splitted_text:
                if word in self.likelihoods:
                    likelihoods_per_word = list(self.likelihoods[word].values())
                    array_of_probabilities = np.vstack((array_of_probabilities, np.array([likelihoods_per_word])))
        
            # To find prediction per class 
            product_of_likelihoods = np.prod(array_of_probabilities, axis=0) # product of likelihoods and priors

            total_probability = np.sum(product_of_likelihoods)
            
            if total_probability == 0:
                # in order to prevent zero division error
                prediction_per_class = np.zeros(shape=product_of_likelihoods.shape)
            else:
                prediction_per_class = product_of_likelihoods/total_probability

            predictions = np.vstack((predictions, prediction_per_class))

        return predictions
    
    def predict_log(self, X:str |np.ndarray | Iterable[str]):
        if isinstance(X, str):
            X = [X]
        
        predictions = np.empty((0, len(self.label_map)))

        for text in X:
            array_of_probabilities = np.empty((0, len(self.label_map)))
            # Adding priors 
            array_of_probabilities = np.vstack((array_of_probabilities, np.array(list(self.priors.values()))))

            splitted_text = set(text.lower().split(' '))
            for word in splitted_text:
                if word in self.likelihoods:
                    likelihoods_per_word = list(self.likelihoods[word].values())
                    array_of_probabilities = np.vstack((array_of_probabilities, np.array([likelihoods_per_word])))

            # To find prediction per class
            log_likelihood = np.log(array_of_probabilities)
            log_sum_of_likelihoods = np.sum(log_likelihood, axis=0)

            predictions = np.vstack((predictions, self.softmax(log_sum_of_likelihoods)))
        return predictions

    def softmax(self, arr:np.ndarray):
        exp_arr = np.exp(arr)
        total = np.sum(exp_arr)
        if total == 0:
            prob = np.zeros(shape=exp_arr.shape)
        else:
            prob = exp_arr/total

        return prob
    
    def score(self, X:np.ndarray|Iterable[str] , y:np.ndarray):
        """Returns the accurcacy score of the model"""
        predictions = self.predict(X)

        indices_of_max_prob = np.argmax(predictions, axis=1)

        compare = indices_of_max_prob == y

        sum_of_correct_predictions = np.sum(compare)

        accuracy = sum_of_correct_predictions/y.size

        return accuracy



if __name__ == "__main__":
    import pandas as pd
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "./emails.csv")

    email = pd.read_csv(path)
    X = email['text'].to_numpy()
    y = email['spam'].to_numpy()

    cls = multinomialNB()
    cls.fit(X, y)
    prediction = cls.predict("click the link below to buy")

    max_index = np.argmax(prediction)

    result_map = {0:'Not spam', 1:'Spam'}

    accuracy = cls.score(X, y)

    print(f"{prediction}, text is {result_map[max_index]}")

    print(accuracy)