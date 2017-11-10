from Ensembles.SoftEnsemble import *
from Validation import cross_validate

# Load Data
xTrain = np.genfromtxt("Data/X_train.txt", delimiter=None)
yTrain = np.genfromtxt("Data/Y_train.txt", delimiter=None)
#xValid = np.genfromtxt("Data/X_test.txt", delimiter=None)


# M: Number of Data
M = xTrain.shape[0]
# N: Number of Features
N = xTrain.shape[1]
# K: Number of Classes
classes = np.unique(yTrain)
K = len(classes)
print("M: %d, N: %d, K: %d, Classes: %s\n" % (M, N, K, classes))


# Set of possible tests to run. These are the parameters for cross_validate.
# You can append a new set to the list rather than erasing what was already there.
# Classifier, X Subset, Y Subset, Cross-Validations
tests = [
    (AClassifier(), xTrain[:5000, :], yTrain[:5000], 20),  # Example 1
    (SoftEnsemble([AClassifier(), AClassifier()]), xTrain, yTrain, 20),  # Example 2
]
index = int(input("Select a test number to run [0,%d]: " % (len(tests)-1)))
if index < 0 or index >= len(tests):
    print("Must select a valid index.")
    quit()


# Calculate the error.
print("Error:", cross_validate(*tests[index]))
