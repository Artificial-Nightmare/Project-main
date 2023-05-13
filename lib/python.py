import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'libsoftware.dll')
sys.path.append(dll_path)

# Load the DLL
dll = ctypes.CDLL(dll_path)


# Définition des types de données d'entrée et de sortie de la fonction
dll.linear_classification.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.c_int, ctypes.c_char_p]
dll.linear_classification.restype = ctypes.POINTER(ctypes.c_double)

# Préparation des données
X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
Y = [1, -1, 1]
X_array = (ctypes.POINTER(ctypes.c_double) * len(X))(*[ctypes.pointer((ctypes.c_double * len(x))(*x)) for x in X])
Y_array = (ctypes.c_int * len(Y))(*Y)
learning_rate = 0.01
max_iterations = 1000
filename = 'resultats.txt'.encode('utf-8')

# Appel de la fonction
resultat = dll.linear_classification(X_array, Y_array, learning_rate, max_iterations, filename)

# Conversion des résultats en un tableau Python
resultat = ctypes.cast(resultat, ctypes.POINTER(ctypes.c_double * (len(X[0]) + 1)))
resultat = list(resultat.contents)

# Affichage de la classification
plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=np.array(Y))
x = np.linspace(0, 10, 100)
y = (-resultat[0] - resultat[1] * x) / resultat[2]
plt.plot(x, y)
plt.show()