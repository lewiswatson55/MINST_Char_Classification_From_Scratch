# MINST Character Classification NN using Pure Python and Numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neuralnetwork as nn

# Load the data
train = np.array(pd.read_csv('mnist/mnist_train.csv'))
m_train, n_train = train.shape
train = train.T  # Transpose the training data
Y_train = train[0]  # The first row is the label
X_train = train[1:]  # The rest is the data
X_train = X_train / 255  # Normalize the data

test = np.array(pd.read_csv('mnist/mnist_test.csv'))
m_test, n_test = test.shape
test = test.T
X_test = test[1:, 100:]
Y_test = test[0, 100:]
X_test = X_test / 255

X_val = test[1:, :100]
Y_val = test[0, :100]
X_val = X_val / 255

# print(X_val.shape)
# print(Y_val.shape)
# print(X_test.shape)
# print(Y_test.shape)
# print(X_train.shape)
# print(Y_train.shape)

lr = 0.2
epochs = 500

w1, b1, w2, b2 = nn.gradient_descent(X_train, Y_train, lr, epochs, X_val, Y_val)
nn.test(w1, b1, w2, b2, X_test, Y_test)  # Test The Model
