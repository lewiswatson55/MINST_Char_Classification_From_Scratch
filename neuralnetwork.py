import numpy as np


def init_nn():
    w1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2


def ReLU(x):
    return np.maximum(0, x)


def ReLU_derivative(x):
    return np.where(x <= 0, 0, 1)


def one_hot(Y):
    one_hot_y = np.zeros((Y.size, Y.max() + 1))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def forward_propagation(w1, b1, w2, b2, X):
    z1 = np.dot(w1, X) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def backward_propagation(z1, a1, z2, a2, w1, w2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dz2 = a2 - one_hot_Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * ReLU_derivative(z1)
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2


def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    return w1, b1, w2, b2


def get_accuracy(a2, Y):
    predictions = np.argmax(a2, axis=0)
    return (predictions == Y).mean()


def gradient_descent(X, Y, learning_rate, iterations):
    w1, b1, w2, b2 = init_nn()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(a2, Y))
    return w1, b1, w2, b2


def test(w1, b1, w2, b2, X, Y):
    print()
    print("====== Training Complete ======")
    print("Validating the model...")
    z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, X)
    print("Test Data Accuracy: ", get_accuracy(a2, Y))
