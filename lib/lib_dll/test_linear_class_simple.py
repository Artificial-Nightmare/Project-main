import ctypes
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'liblinear_classification.dll')
sys.path.append(dll_path)
dll = ctypes.cdll.LoadLibrary(dll_path)

# Définition des types de données pour la fonction rosenblatt
dll.rosenblatt.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
                           np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                           ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                           np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                           ctypes.c_int]

# Exemple d'utilisation
# n'oubliez pas de mettre le dtype=np.float64 !
X = np.array([
      [1, 1],
      [2, 3],
      [3, 3]
], dtype=np.float64)
Y = np.array([
      1,
      -1,
      -1
],dtype=np.float64).flatten()
learning_rate = 0.1
max_iterations = 100
# Centrage et réduction des données
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Créer un tableau de sortie pour les poids entraînés
w = np.zeros((X.shape[1] + 1) * 2, dtype=np.float64)

# Appeler la fonction rosenblatt avec un pointeur vers le tableau de sortie
dll.rosenblatt(X, Y, X.shape[0], X.shape[1], learning_rate, max_iterations, w,2)

for i in range(2):  # Remplacez 2 par le nombre de classes que vous voulez
    print(f"Poids pour la classe {i} : {w[i*(X.shape[1] + 1):(i+1)*(X.shape[1] + 1)]}")

# Affichage des données
plt.scatter(X[0, 0], X[0, 1], color='blue')
plt.scatter(X[1:3,0], X[1:3,1], color='red')
# Affichage de la droite de séparation
x = np.linspace(-1, 4, 100)
for i in range(2):  # Remplacez 2 par le nombre de classes que vous voulez
    w_i = w[i*(X.shape[1] + 1):(i+1)*(X.shape[1] + 1)]
    y = -(w_i[0] * x + w_i[2]) / w_i[1]
    plt.plot(x, y, color='black')
plt.show()
