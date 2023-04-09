import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, X):
        labels = []
        
        for i in range(len(X_test)):
            # compute euclidean distances to other datapoints
            distances = [np.sqrt(np.sum((X_test[i] - X[j])**2)) for j in range(len(X))]
   
            # find most frequent class amongst k nearest neighbours
            indices = np.argsort(distances)[:self.k]
            class_labels = [self.y_train[ind] for ind in indices]
            label = max(class_labels, key = class_labels.count)
            labels.append(label)
            
        return labels