import ctypes
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'SVM.dll')
sys.path.append(dll_path)

# Load the DLL
svm_lib = ctypes.cdll.LoadLibrary(dll_path)

# Définition des types de données pour les fonctions de la DLL
svm_lib.trainSVM.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ctypes.c_size_t,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]
svm_lib.trainSVM.restype = None

svm_lib.predictSVM.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_size_t
]
svm_lib.predictSVM.restype = ctypes.c_int

# Fonction pour entraîner le SVM avec l'algorithme du perceptron
def trainSVM(trainingData, numIterations):
    features = trainingData[:, :-1].astype(np.float64)
    labels = trainingData[:, -1].astype(np.float64)

    # Initialiser les poids et le biais à zéro
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = ctypes.c_double(0.0)

    svm_lib.trainSVM(features, features.shape[0], features.shape[1], weights, ctypes.byref(bias), numIterations)

    return weights, bias.value

# Fonction pour prédire la classe d'un exemple avec le SVM entraîné
def predictSVM(features, weights, bias):
    features = [float(x) for x in features]
    return svm_lib.predictSVM(np.array(features, dtype=np.float64), weights, bias, len(features))


trainingData = np.array([
     [1, 1,1],
      [2, 3,-1],
      [3, 3,-1]
])

# Entraînement du SVM
numIterations = 1000
weights, bias = trainSVM(trainingData, numIterations)

# Création des données de test
testData = np.array([
    [-1.5, -1.5],
    [1.5, 1.5]
])

# Affichage des prédictions pour les données de test
for i in range(testData.shape[0]):
    features = testData[i]
    prediction = predictSVM(features, weights, bias)
    print(f"Features: {features[0]}, {features[1]} => Output: {prediction}")

# Tracer les données d'entraînement
plt.scatter(trainingData[:, 0], trainingData[:, 1], c=trainingData[:, 2], cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')

# Tracer les frontières de décision du SVM
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.array([predictSVM([float(x), float(y)], weights, bias) for x, y in zip(xx.ravel(), yy.ravel())], dtype=np.float64)
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='black', linewidths=1)
plt.show()
