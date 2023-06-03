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
                           np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]

# Exemple d'utilisation
# pas oublier de mettre le dtype=np.float64 !
X = np.array([
      [1, 1],
      [2, 3],
      [3, 3]
], dtype=np.float64)
Y = np.array([
      1,
      -1,
      -1
], dtype=np.float64).flatten()
learning_rate = 0.1
max_iterations = 1000
# Centrage et réduction des données
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Créer un tableau de sortie pour les poids entraînés
w = np.zeros(X.shape[1] + 1, dtype=np.float64)

# Appeler la fonction rosenblatt avec un pointeur vers le tableau de sortie
dll.rosenblatt(X, Y, X.shape[0], X.shape[1], learning_rate, max_iterations, w)


# Affichage des poids
print(w)

# Affichage des données
for i in range(len(X)):
      if Y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='blue')
      else:
            plt.scatter(X[i, 0], X[i, 1], color='red')
x = np.linspace(-5, 4, 100)        
y = (w[0] * x + w[2]) / w[1]
print("X",X,'\n',"Y", y)
print("le poids !",w[0], w[1], w[2])
plt.plot(x, y, 'k-')
plt.show()
plt.clf()
