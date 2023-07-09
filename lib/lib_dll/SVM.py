import ctypes
import numpy as np
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'SVM.dll')
sys.path.append(dll_path)

# Load the DLL
svm_dll = ctypes.cdll.LoadLibrary(dll_path)


# Définition des types de données pour les fonctions de la DLL
svm_dll.trainSVM.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="C_CONTIGUOUS"),
                             ctypes.c_size_t,
                             ctypes.c_size_t,
                             np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.c_int]

svm_dll.predictSVM.restype = ctypes.c_int
svm_dll.predictSVM.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
                               np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
                               ctypes.c_double,
                               ctypes.c_size_t]


def create_training_data():
    trainingData = np.array([
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, -1],
        [0, 0, -1]
    ], dtype=np.double)

    return trainingData

# Entraînement du SVM
def train_svm(trainingData):
    numExamples, numFeatures = trainingData.shape[0], trainingData.shape[1] - 1
    weights = np.zeros(numFeatures, dtype=np.double)
    bias = ctypes.c_double()

    svm_dll.trainSVM(trainingData, numExamples, numFeatures, weights, ctypes.byref(bias), 1000000)

    return weights, bias.value

# Prédiction avec le SVM entraîné
def predict_svm(features, weights, bias):
    prediction = svm_dll.predictSVM(features, weights, bias, len(features))
    return prediction

# Création des données d'entraînement pour la classification linéaire
trainingData = create_training_data()

# Entraînement du SVM
weights, bias = train_svm(trainingData)

# Affichage des prédictions pour toutes les données d'entraînement
for example in trainingData:
    features = example[:-1]
    prediction = predict_svm(features, weights, bias)
    print(f"Features: {features} => Output: {prediction}")
