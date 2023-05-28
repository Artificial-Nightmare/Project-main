import os
from PIL import Image

source_folder = 'chemin_vers_le_dossier_source'
target_folder = 'chemin_vers_le_dossier_cible'
target_size = (400, 400)

if not os.path.exists(""):
    os.makedirs(target_folder)

for filename in os.listdir(source_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        with Image.open(source_path) as image:
            resized_image = image.resize(target_size)
            resized_image.save(target_path)
