import numpy as np
import pandas as pd

class Perceptron(object):

  def __init__(self, learning_rate=0.01, n_epoch=100):
    self.learning_rate = learning_rate
    self.n_epoch = n_epoch
    self.weights = None
    self.bias = None
      
  def train_weights(self, X, y_train):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0
    # Training loop through the entire training data
    for epoch in range(self.n_epoch):
      for idx in range(n_samples):
        output = np.dot(X[idx], self.weights) + self.bias
        y_pred = self.activation_func(output)
        error = y_train[idx] - y_pred
        self.weights += self.learning_rate * error * X[idx]
        self.bias += self.learning_rate * error
      
  def predict(self, X):
    n_samples = len(X)
    y_pred = np.zeros(n_samples)
    for idx in range(n_samples):
      output = np.dot(X[idx], self.weights) + self.bias
      y_pred[idx] = self.activation_func(output)
    return y_pred

  def activation_func(self, output):
    return 1.0 if output >= 0 else 0.0