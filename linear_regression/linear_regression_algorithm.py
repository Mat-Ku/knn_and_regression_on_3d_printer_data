import numpy as np


def mse(y, y_hat):
    return np.mean((y - y_hat)**2)



class LinReg:

    
    def __init__(self, n, alpha):
        self.n = n          # number of iterations
        self.alpha = alpha  # learning rate
        self.w = None       # weights
        self.b = None       # bias


    def fit(self, X_train, y_train):
        
        # Initialize weights and bias as 0
        n_rows, n_columns = X_train.shape
        self.w = np.zeros(n_columns)
        self.b = 0

        # Update parameters across predefined number of iterations
        for i in range(self.n):
            
            # Compute prediction according to y = w * X + b
            y_pred = np.dot(X_train, self.w) + self.b

            # Compute gradient w.r.t. weights and bias
            dw = (1/n_rows) * np.dot(X_train.T, (y_pred - y_train))
            db = (1/n_rows) * np.sum(y_pred - y_train)

            # Update weights and bias
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db


    def predict(self, X_test):
        return np.dot(X_test, self.w) + self.b