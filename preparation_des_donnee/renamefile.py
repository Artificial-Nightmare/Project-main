import os

basket = 'Balle_de_basketball'
football = 'Balle_de_football'
baseball = "Balle_de_baseball"

nameBasketball = "basketball_"
nameFootball = "football_"
nameBaseball = "baseball_"


# Chemin du dossier contenant les images à renommer
dirname = os.path.abspath(os.path.dirname(__file__))
source_dir = os.path.join(dirname,"..", "dataset", baseball)

# Liste des fichiers dans le dossier
files = os.listdir(source_dir)

# Compteur pour numéroter les images
count = 1

# Pour chaque fichier dans le dossier
for file in files:
    # Vérification que le fichier est bien une image
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith('jpeg'):
        # Chemin complet du fichier
        source_path = os.path.join(source_dir, file)
        # Nouveau nom de fichier avec numéro séquentiel
        new_filename = f"{nameBaseball}{count}.png"
        # Chemin complet du nouveau fichier
        dest_path = os.path.join(source_dir, new_filename)
        # Renommage du fichier
        os.rename(source_path, dest_path)
        # Incrément du compteur
        count += 1

print("Les images ont été renommées avec succès !")
