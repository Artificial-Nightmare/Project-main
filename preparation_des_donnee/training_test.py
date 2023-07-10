import os 
from PIL import Image
import numpy as np
import ctypes
import sys
import create_listTest
import trainer
import pickle

dirname = os.path.abspath(os.path.dirname(__file__))
chemin = os.path.join(dirname,"..", "Test_image")
create_listTest.allcolors(chemin)
create_listTest.expected_image(chemin)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'perceptron_multi_couche.dll')
sys.path.append(dll_path)

# Load the DLL
mlp_dll = ctypes.cdll.LoadLibrary(dll_path)

# Définition des types de données
mlp_dll.createMLP.restype = ctypes.c_void_p
mlp_dll.createMLP.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
mlp_dll.deleteMLP.argtypes = [ctypes.c_void_p]

# Chargement des données d'entraînement et de test
# Définition de la structure du MLP

npl = np.array([1875,2048,1024,512,3], dtype=np.int32)
num_epochs = 10000
learning_rate = 0.01045
mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), npl.size)

data_dir = os.path.join(current_dir, '..', 'Test_image')
train_inputs = np.array(create_listTest.allcolors(os.path.join(data_dir, '..', 'Train_image')))
train_expected_outputs = np.array(create_listTest.expected_image(os.path.join(data_dir, '..', 'Train_image')))
print(train_inputs.shape)
# Entraînement du MLP sur un nombre spécifique d'époques
samples_inputs = train_inputs
samples_expected_outputs = train_expected_outputs

mlp_ptr = trainer.training(num_epochs, learning_rate,samples_inputs,samples_expected_outputs,mlp_ptr)


# Charger les données de test
test_inputs = np.array(create_listTest.allcolors(os.path.join(data_dir, '..', 'Test_image')))
test_expected_outputs = np.array(create_listTest.expected_image(os.path.join(data_dir, '..', 'Test_image')))

# Convertir le pointeur mlp_ptr en un pointeur valide de type MLP
mlp = ctypes.cast(mlp_ptr, ctypes.POINTER(ctypes.c_void_p)).contents
print(mlp)

# Charger les images d'entrée et les étiquettes attendues
input_images = test_inputs
expected_outputs = test_expected_outputs

# Appeler la fonction predict_list
predicted_outputs = trainer.predict(mlp_ptr, test_inputs)

# Obtenir l'indice de la classe prédite pour chaque prédiction
# Obtenir l'indice de la classe attendue pour chaque étiquette
predicted_classes = np.argmax(predicted_outputs, axis=1)
expected_classes = np.argmax(test_expected_outputs, axis=1)

# Afficher les classes prédites et les classes attendues côte à côte
print("Prédictions du MLP :")
for i in range(len(predicted_classes)):
    predicted_class = predicted_classes[i]
    expected_class = expected_classes[i]
    print(f"Exemple {i+1} - Prédiction : {predicted_class}, Attendu : {expected_class}")
    print(f"Sortie du MLP : {predicted_outputs[i]}")
