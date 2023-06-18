import os 
from PIL import Image
import numpy as np
import ctypes
import sys

def expected_image(directory):
    expected = []
    for filename in os.listdir(directory):
        if filename.startswith("basketball_"):
            expected.append((1, -1, -1))
        elif filename.startswith("football_"):
            expected.append((-1, 1, -1))
        elif filename.startswith("baseball_"):
            expected.append((-1, -1, 1))
    return np.array(expected)

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
        return np.array(all_pixels)
    else:
        print(f"Aucune image valide trouvée dans le dossier {directory}")



dirname = os.path.abspath(os.path.dirname(__file__))
chemin = os.path.join(dirname,"..", "Test_image")
allcolors(chemin)
expected_image(chemin)


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
mlp_dll.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
mlp_dll.train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_double]

# Création du MLP avec 1875 entrées, 10 neurones cachés et 3 sorties


while True:
    try:
        # Chargement des données d'entraînement et de test
        npl = np.array([1875, 20, 3], dtype=np.int64)
        mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), npl.size)
        data_dir = os.path.join(current_dir, '..', 'Test_image')
        train_inputs = allcolors(os.path.join(data_dir, '..', 'Train_image'))
        train_expected_outputs = expected_image(os.path.join(data_dir, '..', 'Train_image'))
        test_inputs = allcolors(os.path.join(data_dir, '..','Test_image'))
        test_expected_outputs = expected_image(os.path.join(data_dir, '..','Test_image'))

        if train_inputs is not None and train_expected_outputs is not None and test_inputs is not None and test_expected_outputs is not None:
            # Entraînement du MLP sur les données d'entraînement
            mlp_dll.train(mlp_ptr,
                           train_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           train_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           train_inputs.shape[0], train_inputs.shape[1], train_expected_outputs.shape[1],
                           True, 200, 0.00005)

            # Test du MLP sur les données de test
            correct_predictions = 0
            for i in range(test_inputs.shape[0]):
                input = test_inputs[i]
                expected_output = test_expected_outputs[i]
                output = np.zeros(3, dtype=np.double)
                mlp_dll.predict(mlp_ptr,
                                input.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                input.size,
                                True,
                                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                output.size)
                predicted_class = np.argmax(output)
                expected_class = np.argmax(expected_output)
                if predicted_class == expected_class:
                    correct_predictions += 1

            # Affichage de la performance du MLP sur les données de test
            accuracy = correct_predictions / test_inputs.shape[0]
            print(f"Nombre de prédictions correctes : {correct_predictions}")
            print(f"Nombre de prédictions incorrectes : {test_inputs.shape[0] - correct_predictions}")
            print(f"Accuracy : {accuracy}")
            # Suppression du MLP
            mlp_dll.deleteMLP(mlp_ptr)
            break

    except:
        continue


