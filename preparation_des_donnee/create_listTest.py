import os 
from PIL import Image

def expected_image(directory):
    expected = []
    for filename in os.listdir(directory):
        if filename.startswith("basketball_"):
            expected.append((1, 0, 0))
        elif filename.startswith("football_"):
            expected.append((0, 1, 0))
        elif filename.startswith("baseball_"):
            expected.append((0, 0, 1))
    print(expected)

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
            for y in range(25):
                ligne = []
                for x in range(25):
                    r, g, b = image.getpixel((x, y))
                    ligne.append((r, g, b))
                pixels.append(ligne)
            all_pixels.append(pixels)
    if all_pixels:
        print("Liste globale de couleurs de pixels :")
        print(all_pixels)
    else:
        print(f"Aucune image valide trouvée dans le dossier {directory}")


dirname = os.path.abspath(os.path.dirname(__file__))
chemin = os.path.join(dirname,"..", "Test_image")
allcolors(chemin)
expected_image(chemin)