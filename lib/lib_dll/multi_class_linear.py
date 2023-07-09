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
X = np.random.random((500, 2)) * 2.0 - 1.0
Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
              [0, 0, 0]for p in X], dtype=np.float64)

Xp = X.astype(np.float64)
Yp = Y.flatten().astype(np.float64)

learning_rate = 0.001
max_iterations = 20000
# Centrage et réduction des données
Xp = (Xp - np.mean(Xp, axis=0)) / np.std(Xp, axis=0)

# Créer un tableau de sortie pour les poids entraînés
w = np.zeros((Xp.shape[1] + 1) * 3, dtype=np.float64)  # Remplacez 3 par le nombre de classes que vous voulez

# Appeler la fonction rosenblatt avec un pointeur vers le tableau de sortie
dll.rosenblatt(Xp, Yp, Xp.shape[0], Xp.shape[1], learning_rate, max_iterations, w, 3)

for i in range(3):
    print(f"Poids pour la classe {i} : {w[i*(Xp.shape[1] + 1):(i+1)*(Xp.shape[1] + 1)]}")

plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], color='blue')
plt.scatter(X[Y[:, 1] == 1, 0], X[Y[:, 1] == 1, 1], color='red')
plt.scatter(X[Y[:, 2] == 1, 0], X[Y[:, 2] == 1, 1], color='green')

# Affichage des droites de séparation
x = np.linspace(-2, 2, 100)
for i in range(3):
    w_i = w[i*(Xp.shape[1] + 1):(i+1)*(Xp.shape[1] + 1)]
    y = -(w_i[0] * x + w_i[2]) / w_i[1]
    plt.plot(x, y, color='black')

plt.show()
