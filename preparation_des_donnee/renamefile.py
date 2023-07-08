import os



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
source_dir = os.path.join(dirname, "..", data3, basket)

# Liste des fichiers dans le dossier
files = os.listdir(source_dir)

# Liste des numéros de fichiers existants
existing_nums = []
for file in files:
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith('jpeg'):
        num = int(file.replace('.jpg','').replace('.png','').replace('.jpeg','').split('_')[-1])
        existing_nums.append(num)

# Compteur pour numéroter les images
count = 1

# Pour chaque fichier dans le dossier
for file in files:
    # Vérification que le fichier est bien une image
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith('jpeg'):
        # Chemin complet du fichier
        source_path = os.path.join(source_dir, file)
        
        # Saute les numéros de fichiers existants
        while count in existing_nums:
            count += 1
        
        # Nouveau nom de fichier avec numéro séquentiel
        new_filename = f"{nameBasketball}{count}.jpg"
        # Chemin complet du nouveau fichier
        dest_path = os.path.join(source_dir, new_filename)
        # Renommage du fichier
        os.rename(source_path, dest_path)
        # Incrément du compteur
        count += 1

print("Les images ont été renommées avec succès !")