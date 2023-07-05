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



#rezise_all_image(output_folder, new_size=(25, 25))

