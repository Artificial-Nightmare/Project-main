import numpy as np
import matplotlib.pyplot as plt
from typing import List

class MyMLP:
  # Constructeur
  # npl : Neurons Per Layer - représentant la structure du PMC choisie par l'utilisateur 
  # (nombre d'entrées, nombre de neurones par couches cachées, nombre de sorties)
  def __init__(self, npl: List[int]):
    self.d = npl
    self.L = len(npl) - 1
    self.W = []

    # Initialisation des poids du modèles entre -1 et 1 (sauf pour les poids inutiles que l'on laissera à 0)
    for l in range(self.L + 1):
      self.W.append([])

      if l == 0:
        continue
      for i in range(npl[l - 1] + 1):
        self.W[l].append([])
        for j in range(npl[l] + 1):
          self.W[l][i].append(0.0 if j == 0 else np.random.uniform(-1.0, 1.0))
    
    # Création de l'espace mémoire pour 'stocker' plus tard les valeurs de sorties de chaque neurone
    self.X = []
    for l in range(self.L + 1):
      self.X.append([])
      for j in range(npl[l] + 1):
        self.X[l].append(1.0 if j == 0 else 0.0)

    # Création de l'espace mémoire pour 'stocker' plus tard les semi-gradient associés à chaque neurone
    self.deltas = []
    for l in range(self.L + 1):
      self.deltas.append([])
      for j in range(npl[l] + 1):
        self.deltas[l].append(0.0)

  # Propagation et mise à jour des valeurs de sorties de chaque neurone à partir des entrées d'un exemple
  def _propagate(self, inputs: List[float], is_classification: bool):
    # copie des entrées dans la 'couche d'entrée' du modèle
    for j in range(self.d[0]): 
      self.X[0][j + 1] = inputs[j]
    
    # mise à jour récursive des valeurs de sorties des neurones, couche après couche
    for l in range(1, self.L + 1):
      for j in range(1, self.d[l] + 1):
        total = 0.0
        for i in range(0, self.d[l - 1] + 1):
          total += self.W[l][i][j] * self.X[l - 1][i]
        
        if l < self.L or is_classification:
          total = np.tanh(total)
        
        self.X[l][j] = total

  # Méthode à utiliser pour interroger le modèle (inférence)
  def predict(self, inputs: List[float], is_classification: bool):
    self._propagate(inputs, is_classification)
    return self.X[self.L][1:]

  # Méthode à utiliser pour entrainer le modèle à partir d'un dataset étiqueté
  def train(self, all_samples_inputs: List[List[float]], all_samples_expected_outputs: List[List[float]],
            is_classification: bool, iteration_count: int, alpha: float):
    # Pour un certain nombre d'itération
    for it in range(iteration_count):
      # Choix d'un exemple étiqueté au hasard dans le dataset
      k = np.random.randint(0, len(all_samples_inputs))
      inputs_k = all_samples_inputs[k]
      y_k = all_samples_expected_outputs[k]

      # Mise à jour des valeurs de sorties des neurones du modèle à partir des entrées de l'exemple sélectionné
      self._propagate(inputs_k, is_classification)

      # Calcul des semi gradients des neurones de la dernière couche
      for j in range(1, self.d[self.L] + 1):
        self.deltas[self.L][j] = (self.X[self.L][j] - y_k[j - 1])
        if is_classification:
          self.deltas[self.L][j] *= (1 - self.X[self.L][j] ** 2)

      # Calcul de manière récursive des semi gradients des neurones des couches précédentes
      for l in reversed(range(1, self.L + 1)):
        for i in range(1, self.d[l - 1] + 1):
          total = 0.0
          for j in range(1, self.d[l] + 1):
            total += self.W[l][i][j] * self.deltas[l][j]
          self.deltas[l-1][i] = (1 - self.X[l-1][i] ** 2) * total

      # Correction des poids du modèle
      for l in range(1, self.L + 1):
        for i in range(0, self.d[l - 1] + 1):
          for j in range(1, self.d[l] + 1):
            self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j]

#/////////////////////////////////////////////////////////////////////////////////////////////////////


# Génération de données pour le problème de classification
def generate_data(n_samples: int):
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = (X[:,0] ** 2 + X[:,1] ** 2) < 0.5
    return X, y.astype(int)

# Création du modèle de réseau de neurones
mlp = MyMLP([2, 5, 1])

# Génération des données d'entraînement
X_train, y_train = generate_data(1000)

# Entraînement du modèle
mlp.train(X_train.tolist(), y_train.reshape(-1, 1).tolist(), True, 1000, 0.1)

# Génération de données pour la visualisation
xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
X_test = np.c_[xx.ravel(), yy.ravel()]
y_test = mlp.predict(np.array(X_test.tolist()), True)

# Affichage de la frontière de décision
plt.contourf(xx, yy, y_test.reshape(xx.shape), cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k')
plt.show()
