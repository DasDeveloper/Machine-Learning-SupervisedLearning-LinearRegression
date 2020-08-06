import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("exemple1.csv")

x = data.x.tolist()
X = np.array(x).reshape((len(x), 1))
matrix_ones = np.ones((len(X), 1))
X_b = np.append(matrix_ones, X, axis=1)
y = data.y.tolist()
Y = np.array(y).reshape((len(y), 1))
theta = np.random.randn(2, 1)
m = len(x)


def gradientDescent_batch(X, y, theta, learning_rate=0.01, iterations=10000):
    J_History = np.zeros((iterations, 1))
    theta_history = np.zeros((iterations, 2))

    def computeCost(theta, X, y):
        m = len(y)
        prediction = X.dot(theta)
        cost = 1 / 2 * m * np.sum((np.square(prediction - y)))
        return cost

    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * X.T.dot(prediction - y)
        theta_history[it, :] = theta.T
        J_History[it] = computeCost(theta, X, y)

    return f'Theta0 = {theta[0][0]}, theta1 = {theta[0][1]}', J_History, theta_history


def NormalEquationMethod(X, y):
    X_transpose = X.T
    temp = X_transpose.dot(X)
    inverse = np.linalg.inv(temp)
    temp2 = inverse.dot(X_transpose)
    theta = temp2.dot(y)
    return f'Theta0 = {theta[0][0]}, theta1 = {theta[0][1]}'


def plotData(dataset, a, b):
    x = dataset.x.tolist()
    y = dataset.y.tolist()
    y_linear = []
    for value in range(len(x)):
        y_linear.append(a * x[value] + b)
    plt.plot(x, y, linestyle="none", marker=".")
    plt.xlabel("population of a city in 10 000's ")
    plt.ylabel("the profit of a food truck in that city in 10 000's $")
    plt.plot(x, y_linear, linestyle="dotted")
    plt.show()


