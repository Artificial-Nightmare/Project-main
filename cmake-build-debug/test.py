import matplotlib.pyplot as plt
import numpy as np

# Lecture des données de frontière de décision depuis le fichier texte
with open("data.txt") as f:
    line = f.readline()
    w_values = [float(x) for x in line.split(",")]

# Création de la frontière de décision
x_points = np.linspace(-5,5)
y_points = - (w_values[0] + w_values[1]*x_points) / w_values[2]

# Tracé des points de données et de la frontière de décision
plt.scatter([1,2,3], [1,3,3], c=[1,-1,-1])
plt.plot(x_points, y_points)

# Affichage de la figure
plt.show()
