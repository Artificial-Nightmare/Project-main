import os
from PIL import Image

basket = 'Balle_de_basketball'
football = 'Balle_de_football'
baseball = "Balle_de_baseball"

nameBasketball = "basketball_"
nameFootball = "football_"
nameBaseball = "baseball_"

data1 = "dataset"
data2 = "dataset_copy"
data3 = "dataset_Same_Size"

# Chemin du dossier contenant les images à renommer
dirname = os.path.abspath(os.path.dirname(__file__))
source_dir = os.path.join(dirname,"..", data3, football)

# Liste de tous les fichiers dans le dossier
files = os.listdir(source_dir)

# Pour chaque fichier dans le dossier
for file in files:
    # Vérification que le fichier est bien une image
    if file.endswith(".png"):
        # Chemin complet du fichier
        image_path = os.path.join(source_dir, file)
        # Ouverture de l'image et affichage de sa taille en pixels
        with Image.open(image_path) as img:
            print(f"La taille de l'image {file} est: {img.size}")
        # Fermeture de l'image en dehors du bloc "with"
        img.close()
