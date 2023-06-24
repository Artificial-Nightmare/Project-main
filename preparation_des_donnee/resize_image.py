from PIL import Image
import os

basket = "Balle_de_basketball"
football = "Balle_de_football"
baseball = "Balle_de_baseball"

def resize_image(image_path, output_path, size=(200, 200)):
    with Image.open(image_path) as img:
        if img.mode == "CMYK":
            img = img.convert("RGB")
        img_resized = img.resize(size)
        img_resized.save(output_path)
        # Fermeture de l'image redimensionnée
        img_resized.close()

def know_all_size(directory):
    # Liste de tous les fichiers dans le dossier
    files = os.listdir(directory)
    # Pour chaque fichier dans le dossier
    for file in files:
        # Vérification que le fichier est bien une image
        if file.endswith(".jpg"):
            # Chemin complet du fichier
            image_path = os.path.join(directory, file)
            # Ouverture de l'image et affichage de sa taille en pixels
            with Image.open(image_path) as img:
                print(f"La taille de l'image {file} est: {img.size}")
            # Fermeture de l'image en dehors du bloc "with"
            img.close()

def rezise_all_image(directory, x, y):
    # Liste de tous les fichiers dans le dossier
    files = os.listdir(directory)
    # Pour chaque fichier dans le dossier
    for file in files:
        # Vérification que le fichier est bien une image
        if file.endswith(".jpg"):
            # Chemin complet du fichier
            image_path = os.path.join(directory, file)
            # Ouverture de l'image et affichage de sa taille en pixels
            with Image.open(image_path) as img:
                img_resized = img.resize((x, y))
                img_resized.save(image_path)
                # Fermeture de l'image redimensionnée
                img_resized.close()

dirname = os.path.abspath(os.path.dirname(__file__))
images_folder = os.path.join(dirname, "..", "dataset", basket)
output_folder = os.path.join(dirname, "..", "dataset_Same_Size", basket)
new_size = (25, 25) # Taille de la nouvelle image

# Parcourez chaque image dans le dossier images_folder
for filename in os.listdir(images_folder):
    # Obtenez le chemin complet de l'image originale et de la nouvelle image redimensionnée
    image_path = os.path.join(images_folder, filename)
    output_path = os.path.join(output_folder, filename)
    # Redimensionner l'image
    resize_image(image_path, output_path, size=new_size)

know_all_size(output_folder)

#rezise_all_image(output_folder, new_size=(25, 25))

