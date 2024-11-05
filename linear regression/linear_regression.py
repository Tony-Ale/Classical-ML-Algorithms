import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            error = y_pred - y
            dw = 2 * (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


def r2_score(y_true, y_pred):
    # Mean of actual values
    y_mean = np.mean(y_true)
    
    # Total sum of squares (variance of the data)
    total_sum_of_squares = np.sum((y_true - y_mean) ** 2)
    
    # Residual sum of squares (variance of prediction errors)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    
    # R-squared formula
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2

def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt


    model = LinearRegression(lr=0.001)
    
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    print(f"r2 score is {r2}")

    pred_line = model.predict(X)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s= 10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, pred_line, color="black", linewidth=2, label="prediction")
    plt.show()

if __name__ == "__main__":
    main()

