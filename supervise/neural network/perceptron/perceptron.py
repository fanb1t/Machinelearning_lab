import numpy as np
import matplotlib.pyplot as plt

def train_weights(X, y_train, learning_rate=0.01, n_epoch=100):
  n_input = X.shape[1]
  weights = np.zeros(n_input)
  bias = 0
    # Training loop through the entire training data
  for epoch in range(n_epoch):
    for idx in range(len(X)):
      output = np.dot(X[idx], weights) + bias
      y_pred = activation_func(output)
      error = y_train[idx] - y_pred
      weights += learning_rate * error * X[idx]
      bias += learning_rate * error

    print(f"Epoch {epoch}: weights={weights}, bias={bias}, error={error}")
  return weights, bias

def predict(sample_unit, weights, bias):
  output = np.dot(sample_unit, weights) + bias
  y_pred = activation_func(output)
  return y_pred

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return np.exp(x)/(1+np.exp(x))

# Step function
def activation_func(output):
  return 1.0 if output >= 0 else 0.0

# 1. Training process
X_train = np.array([
  [5.1, 3.5],
  [4.9, 3.0],
  [4.7, 3.2],
  [7.0, 3.2],
  [6.4, 3.2],
  [6.9, 3.1]
])
y_train = np.array([0, 0, 0, 1, 1, 1])

X = X_train
w, b = train_weights(X, y_train, learning_rate=0.01, n_epoch=20)

# 2. Plot decision boundary
# w_0*x_0 + w_1*x_1 + b = 0
x_0 = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 20)
x_1 = -(w[0] / w[1]) * x_0 - (b / w[1])


# 3. testing process
x_test = np.array([
  [4.9, 3.1],
  [5.8, 4.0],
  [6.8, 2.8],
  [6.7, 3.0]
])

y_test = np.array([0, 0, 1, 1])
y_pred = np.array([predict(x, w, b) for x in x_test])

print("ค่าpredicted:", y_pred)
print("ค่าactual:", y_test)

# plt.xlabel("Sepal.Length")
# plt.ylabel("Sepal.Width")
# plt.title("Decision boundary")

# plt.scatter(X[:, 0], X[:, 1], marker="o", c=y_train)
# plt.plot(x_0, x_1, 'b--')
# plt.show()
