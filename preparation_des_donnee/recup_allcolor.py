from PIL import Image
import os

def allcolors(directory):
    # Initialiser la liste globale pour stocker les couleurs de tous les pixels de toutes les images
    all_pixels = []

    # Parcourir tous les fichiers d'images dans le dossier
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):

            # Spécifier le chemin d'accès complet pour l'image
            image_path = os.path.join(directory, filename)

            # Ouvrir l'image
            image = Image.open(image_path)

            # Vérifier et convertir l'image si nécessaire
            if image.mode != 'RGB':
                print(f"L'image {filename} n'est pas en format RGB (mode {image.mode})")
                image = image.convert('RGB')

            # Récupérer la taille de l'image
            #largeur, hauteur = image.size

            # Créer une liste pour stocker les couleurs de chaque pixel de l'image
            pixels = []

            # Parcourir tous les pixels de l'image et stocker les couleurs
            for y in range(25):
                ligne = []
                for x in range(25):
                    r, g, b = image.getpixel((x, y))
                    ligne.append((r, g, b))
                pixels.append(ligne)

            # Ajouter la liste de couleurs de pixels de cette image à la liste globale de toutes les images
            all_pixels.append(pixels)

    # Vérifier le contenu de la liste globale de tous les pixels
    if all_pixels:
        print("Liste globale de couleurs de pixels :")
        print(all_pixels)
        # Accéder aux couleurs des pixels pour l'image numéro 2 (index 1) et le pixel (x=10, y=20)
        if len(all_pixels) > 1 and len(all_pixels[1]) > 20 and len(all_pixels[1][20]) > 10:
            couleur_pixel_2_10_20 = all_pixels[1][20][10]
            print(
                f"Couleurs pixel (x={10}, y={20}) de l'image numéro 2 : {couleur_pixel_2_10_20}"
            )
        else:
            print("Impossible d'accéder au pixel (x=10, y=20) de l'image numéro 2.")
    else:
        print(f"Aucune image valide trouvée dans le dossier {directory}")


# Spécifier le chemin d'accès complet pour le dossier contenant les images
dirname = os.path.abspath(os.path.dirname(__file__))
directory = os.path.join(dirname, "..", "dataset_Same_Size", "Balle_de_basketball")

# Appeler la fonction allcolors() pour récupérer les couleurs de tous les pixels de toutes les images
allcolors(directory)


