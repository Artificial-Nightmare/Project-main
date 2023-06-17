import os 
from PIL import Image
import random

basket = 'Balle_de_basketball'
football = 'Balle_de_football'
baseball = "Balle_de_baseball"

nameBasketball = "basketball_"
nameFootball = "football_"
nameBaseball = "baseball_"


# Spécifier le chemin d'accès complet pour le dossier contenant les images
dirname = os.path.abspath(os.path.dirname(__file__))
images_basket = os.path.join(dirname, "..", "dataset_Same_Size", basket)
images_football = os.path.join(dirname, "..", "dataset_Same_Size", football)
images_baseball = os.path.join(dirname, "..", "dataset_Same_Size", baseball)

number_images = 5 

all_images_basket = os.listdir(images_basket)
all_images_football = os.listdir(images_football)
all_images_baseball = os.listdir(images_baseball)

images_basket_random = random.sample(all_images_basket, number_images)
images_football_random = random.sample(all_images_football, number_images)
images_baseball_random = random.sample(all_images_baseball, number_images)

all_