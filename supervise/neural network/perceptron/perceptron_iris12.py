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

data = pd.read_csv("D:/learnprogrammer/Machine_learning/deep learning/neural network/perceptroniris12_test.csv")
df = data[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
'Petal.Width']]
labels = data[['Species']]
X_train = df.values
y_train = np.where(labels.values == 'versicolor', 0,1)
p = Perceptron(learning_rate=0.02, n_epoch=10)
p.train_weights(X_train, y_train)

test_data = pd.read_csv("D:/learnprogrammer/Machine_learning/deep learning/neural network/perceptroniris12_test.csv")
x_test = test_data[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
'Petal.Width']].values
y_pred = p.predict(x_test)
y_test = np.where(test_data['Species'].values == 'versicolor', 0, 1)
print("Predicted:", y_pred)
print("Actual:", y_test)

def evaluate_model(y_true, y_pred):
    correct = 0
    total = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
        total += 1
    accuracy = correct / total
    return accuracy