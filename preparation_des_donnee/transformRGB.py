
from PIL import Image
import os

data1 = "Balle_de_basketball"
data2 = "Balle_de_football"
data3 = "Balle_de_baseball"


# Chemin du dossier contenant les images
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset_Same_Size", data3))

# Parcourir tous les fichiers dans le dossier
for filename in os.listdir(dir_path):
    # Chemin complet du fichier
    file_path = os.path.join(dir_path, filename)
    
    # Vérifier si le fichier est une image
    if file_path.endswith(".jpg"):
        # Ouvrir l'image avec PIL
        image = Image.open(file_path)
        
        # Vérifier si l'image est dans un format à convertir
        if image.mode in ["RGBA", "L", "P"]:
            # Convertir l'image en mode RGB
            image = image.convert("RGB")
            # Enregistrer l'image convertie
            image.save(file_path)
            print(f"{file_path} a été converti en mode RGB")