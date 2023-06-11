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
X_ctype = X.astype(np.float64)
Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
              [0, 0, 0]for p in X], dtype=np.float64)
Y_ctype = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
              [0, 0, 0]for p in X], dtype=np.float64).flatten()
learning_rate = 0.00004
max_iterations = 5000000
nb_classes = 3
W = np.random.random((X.shape[1] + 1) * nb_classes) - 0.5


dll.rosenblatt(X_ctype, Y_ctype, X_ctype.shape[0], X.shape[1], learning_rate, max_iterations, nb_classes, W)

# Afficher les poids pour chaque perceptron
W = W.reshape((X.shape[1] + 1, nb_classes))
print(W)


# Prédiction de la classe pour chaque observation
predictions = np.argmax(np.dot(np.concatenate([X, np.ones((X.shape[0], 1))], axis=1), W), axis=1)

# Affichage des données
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1], color='blue')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1], color='red')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1], color='green')

# Affichage de la droite de séparation pour chaque perceptron
x = np.linspace(-1.5, 1.5, 1000000)
c = 0
w = W[:-1, c]
b = W[-1, c]
x_min, x_max = plt.xlim()
x_values = np.linspace(x_min, x_max, 2)
while c < nb_classes: 
    w = W[:-1, c]
    b = W[-1, c]
    y_values = -(w[1] / w[0]) * (x_values - b / w[0])
    plt.plot(x_values, y_values)
    c += 1

plt.show()