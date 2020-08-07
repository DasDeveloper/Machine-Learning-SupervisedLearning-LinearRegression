from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Initialize some data
data = pd.read_csv("example2.csv")
n = 2
x1 = data.x1.tolist()
x2 = data.x2.tolist()


# Feature scaling and mean normalization Note: gradientDescent doesn't work without those.
def featureScaling_mean_normalization(dataset):
    biggest_number = max(dataset)
    smallest_number = min(dataset)
    range_value = biggest_number - smallest_number
    average = sum(dataset)/len(dataset)
    for iterations in range(len(dataset)):
        (dataset[iterations]-average) / range_value


featureScaling_mean_normalization(x1)
featureScaling_mean_normalization(x2)

##################################################
y = data.y.tolist()
m = len(x1)
theta = np.random.randn(n + 1, 1)  # Used for gradient descent only
matrix_one = np.ones(len(x1))
X1 = np.array(x1).reshape(m, 1)
X2 = np.array(x2).reshape(m, 1)

# Creating the matrices
X = np.column_stack((matrix_one, X1, X2))
Y = np.array(y).reshape(m, 1)
J_History = []
Iterations = []


def normalEquation(X, y):
    X_transpose = X.T
    temp = X_transpose.dot(X)
    inverse = np.linalg.inv(temp)
    temp2 = inverse.dot(X_transpose)
    theta = temp2.dot(y)
    return theta


def computeCost(X, Y, theta):
    prediction = np.dot(X, theta)
    J = 1 / (2 * m) * np.sum(np.square(prediction - Y))
    return J


def gradientDescent(X, Y, theta, iterations=100, learning_rate=0.0000001):
    for it in range(iterations):
        Iterations.append(it)
        J_History.append(computeCost(X, Y, theta))
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * X.T.dot(prediction - Y)
    plt.plot(Iterations, J_History)
    plt.xlabel("iterations")
    plt.ylabel("J(theta)")
    plt.show()
    return theta

print("Gradient")
print(gradientDescent(X, Y, theta))
print("Normal :")
print(normalEquation(X, Y))
