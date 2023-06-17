import os
from PIL import Image

# Spécifier le chemin d'accès complet pour le dossier contenant les images
dirname = os.path.abspath(os.path.dirname(__file__))
images_folder = os.path.join(dirname, "..", "dataset_Same_Size", "Balle_de_baseball")

# Spécifier le nom du fichier image
filename = "baseball_1.jpg"

# Combiner les chemins d'accès pour atteindre l'image
image_path = os.path.join(images_folder, filename)

# Vérifier si le fichier image existe
if not os.path.exists(image_path):
    print("Le fichier image {} n'existe pas".format(image_path))
else:
    # Ouvrir l'image
    image = Image.open(image_path)

    # Récupérer la taille de l'image
    largeur, hauteur = image.size

    # Créer une liste pour stocker les couleurs de chaque pixel
    pixels = []

    # Parcourir tous les pixels de l'image
    for y in range(hauteur):
        ligne = []
        for x in range(largeur):
            r, g, b = image.getpixel((x, y))
            ligne.append((r, g, b))
        pixels.append(ligne)

    # Afficher les couleurs
    for y in range(hauteur):
        for x in range(largeur):
            r, g, b = pixels[y][x]
            print("Pixel ({}, {}) : R = {}, G = {}, B = {}".format(x, y, r, g, b))