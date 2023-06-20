import os 
from PIL import Image
import numpy as np
import ctypes
import sys

def expected_image(directory):
    expected = []
    for filename in os.listdir(directory):
        if filename.startswith("basketball_"):
            expected.append(0)
        elif filename.startswith("football_"):
            expected.append(1)
        elif filename.startswith("baseball_"):
            expected.append(2)
        else:
            print(f"Le fichier {filename} n'a pas été étiqueté car son nom ne commence pas par basketball_, football_ ou baseball_.")
    
    # Convertir les étiquettes en one-hot encoding
    num_classes = len(set(expected))
    one_hot_expected = np.zeros((len(expected), num_classes))
    for i, val in enumerate(expected):
        one_hot_expected[i][val] = 1
   
    return np.array(one_hot_expected, dtype=np.double)



def allcolors(directory):
    all_pixels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                print(f"L'image {filename} n'est pas en format RGB (mode {image.mode})")
                image = image.convert('RGB')
            pixels = []
            for pixel in image.getdata():
                r, g, b = pixel
                pixels.append((r, g, b))
            all_pixels.append(pixels)
    if all_pixels:
        all_pixels = np.array(all_pixels)
        all_pixels = (all_pixels - np.mean(all_pixels, axis=0)) / np.std(all_pixels, axis=0)
        return np.array(all_pixels,dtype=np.double)
    else:
        print(f"Aucune image valide trouvée dans le dossier {directory}")







