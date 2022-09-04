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
Y_test = test[0]
X_test = test[1:]
X_test = X_test / 255


print(train.shape)
print(test.shape)
print(m_test)
print(m_train)


w1, b1, w2, b2 = nn.gradient_descent(X_train, Y_train, 0.5, 500)
nn.test(w1, b1, w2, b2, X_test, Y_test)