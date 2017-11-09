import numpy as np

from Classifiers.AClassifier import AClassifier
from Validation import *

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


# [Classifiers], X Subset, Y Subset, Cross-Validations
tests = [
    ([AClassifier(), AClassifier()], xTrain[:, :], yTrain[:], 20),
    # Your Tests Here
]
index = int(input("Select a test number to run [0,%d]:" % (len(tests)-1)))
if index < 0 or index >= len(tests):
    print("Must select a valid index.")
    quit()


# Produce the error
print("Error:", cross_validate(*tests[index]))
