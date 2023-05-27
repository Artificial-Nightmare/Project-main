import ctypes
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'perceptron_multi_couche.dll')
sys.path.append(dll_path)

# Charger la DLL
pmc_lib = ctypes.CDLL(dll_path)

# Définir les types des arguments et du résultat de la fonction train
from ctypes import *
pmc_lib.train.argtypes = [
    c_void_p, # PMC instance
    POINTER(c_double), # entrées : tableau de nombres à virgule flottante
    POINTER(c_double), # sorties attendues : tableau de nombres à virgule flottante
    c_int, # nombre d'échantillons
    c_int, # nombre d'entrées
    c_int, # nombre de sorties
    c_bool, # booléen indiquant si le problème est de classification (True) ou de régression (False)
    c_int, # nombre d'itérations
    c_double, # taux d'apprentissage
]
pmc_lib.train.restype = None # La fonction train ne renvoie rien

# Définir les types des arguments et du résultat de la fonction predict
pmc_lib.predict.argtypes = [
    c_void_p, # PMC instance
    POINTER(c_double), # entrées : tableau de nombres à virgule flottante
    c_int, # nombre d'entrées
]
pmc_lib.predict.restype = POINTER(c_double) # La fonction predict renvoie un tableau de nombres à virgule flottante

# Définir la classe MyMLP et ses méthodes
class MyMLP(ctypes.Structure):
    pass

MyMLP_p = ctypes.POINTER(MyMLP)

pmc_lib.create_mlp.restype = MyMLP_p
pmc_lib.create_mlp.argtypes = [
    c_int, # nombre de couches cachées
    POINTER(c_int), # tableau des tailles des couches cachées
    c_int, # nombre d'entrées
    c_int, # nombre de sorties
    c_bool, # booléen indiquant si le problème est de classification (True) ou de régression (False)
    c_double, # borne inférieure pour l'initialisation des poids
    c_double, # borne supérieure pour l'initialisation des poids
    c_double, # taux de régularisation des poids
    c_int, # fonction d'activation : 0 = sigmoïde, 1 = tangente hyperbolique
]

def create_mlp(hidden_layers_sizes, num_inputs, num_outputs, is_classification):
    num_hidden_layers = len(hidden_layers_sizes)
    hidden_layers_sizes_array = (ctypes.c_int * num_hidden_layers)(*hidden_layers_sizes)
    mlp = pmc_lib.create_mlp(num_hidden_layers, hidden_layers_sizes_array, num_inputs, num_outputs, is_classification, -0.5, 0.5, 0.0, 0)
    return mlp

def train(mlp, inputs, expected_outputs, num_samples, num_inputs, num_outputs, is_classification, num_iterations, learning_rate):
    inputs_p = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    expected_outputs_p = expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pmc_lib.train(mlp, inputs_p, expected_outputs_p, num_samples, num_inputs, num_outputs, is_classification, num_iterations, learning_rate)

def predict(mlp, inputs, num_inputs):
    inputs_p = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output_array = pmc_lib.predict(mlp, inputs_p, num_inputs)
    output_list = [output_array[i] for i in range(num_inputs)]
    return output_list

# Définir les données d'entraînement pour le PMC
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
expected_outputs = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Créer une instance de la classe MyMLP
mlp = create_mlp([3], 2, 1, True)

# Entraîner le PMC
train(mlp, inputs, expected_outputs, 4, 2, 1, True, 1000, 0.1)

# Utiliser le PMC pour prédire les sorties pour de nouvelles entrées
new_inputs = np.array([[0, 1], [1, 0]], dtype=np.float64)
output_list = predict(mlp, new_inputs, 2)

# Afficher les résultats
print('Les sorties prédites pour les nouvelles entrées sont :', output_list)

# Tracer la frontière de décision du PMC
x_min, x_max = -0.1, 1.1
y_min, y_max = -0.1, 1.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
X = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
Z = np.array([predict(mlp, x, 2)[0] for x in X], dtype=np.float64)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(inputs[:, 0], inputs[:, 1], c=expected_outputs[:, 0], cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
