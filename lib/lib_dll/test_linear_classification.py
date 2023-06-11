import ctypes
import numpy as np
import os
import matplotlib.pyplot as plt

# Charger la DLL contenant la fonction rosenblatt
dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'liblinear_classification.dll')
dll = ctypes.cdll.LoadLibrary(dll_path)

# Définir les types de données pour la fonction rosenblatt
dll.rosenblatt.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
                           np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_double,
                           ctypes.c_int,
                           ctypes.c_int,
                           np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]

# Exemple d'utilisation
X = np.random.random((500, 2)) * 2.0 - 1.0
X_bias = np.ones((X.shape[0], 1))
X_ctype = np.hstack((X_bias, X)).astype(np.float64)
Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
              [0, 0, 0]for p in X], dtype=np.float64)
Y_ctype = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
              [0, 0, 0]for p in X], dtype=np.float64).flatten()
learning_rate = 0.01
max_iterations = 50000
nb_classes = 3
W = np.random.normal(0, 1/np.sqrt(X_ctype.shape[1]), size=(X_ctype.shape[1], nb_classes))



dll.rosenblatt(X_ctype, Y_ctype, X_ctype.shape[0], X_ctype.shape[1], learning_rate, max_iterations, nb_classes, W.flatten())


# Afficher les poids pour chaque perceptron
print(W)

# Affichage des données
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1], color='blue')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1], color='red')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1], color='green')


# Affichage de la droite de séparation pour chaque perceptron
x = np.linspace(-1.5, 1.5, 100)
colors = ['blue', 'red', 'green']
for c in range(nb_classes):
    w = W[:-1, c]
    b = W[-1, c]
    y = -(w[0] * x + b) / (w[1] + 1e-10)
    plt.plot(x, y, color=colors[c])

plt.show()