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