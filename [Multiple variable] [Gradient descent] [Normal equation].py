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
    new_dataset = []
    biggest_number = max(dataset)
    smallest_number = min(dataset)
    range_value = biggest_number - smallest_number
    average = sum(dataset)/len(dataset)
    for iterations in range(len(dataset)):
        temp = (dataset[iterations]-average) / range_value
        new_dataset.append(temp)
    return new_dataset


X_1 = featureScaling_mean_normalization(x1)
X_2 = featureScaling_mean_normalization(x2)

##################################################
y = data.y.tolist()
m = len(x1)
theta = np.random.randn(n + 1, 1)  # Used for gradient descent only
matrix_one = np.ones(len(x1))
#Gradient Descent
X1 = np.array(X_1).reshape(m, 1)
X2 = np.array(X_2).reshape(m, 1)

#Normal equation
X1_normal = np.array(x1).reshape(m, 1)
X2_normal = np.array(x2).reshape(m, 1)

# Creating the matrices
#Gradient descent
X = np.column_stack((matrix_one, X1, X2))
Y = np.array(y).reshape(m, 1)

#Normal_equation
X_normal = np.column_stack((matrix_one, X1_normal, X2_normal))
Y_normal = np.array(y).reshape(m, 1)
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


def gradientDescent(X, Y, theta, iterations=1000000, learning_rate=0.03):
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * X.T.dot(prediction - Y)
    # plt.plot(Iterations, J_History)
    # plt.xlabel("iterations")
    # plt.ylabel("J(theta)")
    # plt.show()
    return theta

print("Gradient: ")
print(gradientDescent(X, Y, theta))

print("Normal: ")
print(normalEquation(X_normal, Y_normal))
