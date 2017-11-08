import numpy as np

# Load Data
xTrain = np.genfromtxt("Data/X_train.txt", delimiter=None)
yTrain = np.genfromtxt("Data/Y_train.txt", delimiter=None)
xValid = np.genfromtxt("Data/X_test.txt", delimiter=None)

# M: Number of Data
M = xTrain.shape[0]

# N: Number of Features
N = xTrain.shape[1]

# K: Number of Classes
classes = np.unique(yTrain)
K = len(classes)


# TODO: Generate Predictions


