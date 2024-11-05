import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, lr=0.001, epoch=1000):
        self.lr = lr
        self.epoch = epoch
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epoch):
            linear_model = np.dot(X, self.weights) + self.bias

            y_pred = self.sigmoid(linear_model)


            error = y_pred - y 

            dw = 2*(1/n_samples) * np.dot(error, X)

            db = 2*(1/n_samples) * np.sum(error)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    
    def predict(self, x):
        linear_model = np.dot(x, self.weights) + self.bias

        y_pred = self.sigmoid(linear_model)

        predictions = [0 if y<=0.5 else 1 for y in y_pred]

        return predictions


 
    
    def sigmoid(self, x):
        # Cap the value of x to avoid overflow in exp
        x = np.clip(x, -709, 709) # np.exp(709) is around the max float
        return 1/(1 + np.exp(-x))


def main():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    model =LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    eval = accuracy(y_pred, y_test)
    print(f"Accuracy is {eval}")


def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

if __name__ == "__main__":
    main()






