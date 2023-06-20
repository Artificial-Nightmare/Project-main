import os 
from PIL import Image
import numpy as np
import ctypes
import sys
import create_listTest

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
mlp_dll.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
mlp_dll.train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_double]

# Création du MLP avec 1875 entrées, 10 neurones cachés et 3 sorties

letsgo = 5
# Chargement des données d'entraînement et de test
npl = np.array([1875, 468, 117, 3], dtype=np.int64)
mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), npl.size)
data_dir = os.path.join(current_dir, '..', 'Test_image')
train_inputs = np.array(create_listTest.allcolors(os.path.join(data_dir, '..', 'Train_image')))
train_expected_outputs = np.array(create_listTest.expected_image(os.path.join(data_dir, '..', 'Train_image')))
test_inputs = np.array(create_listTest.allcolors(os.path.join(data_dir, '..','Test_image')))
test_expected_outputs = np.array(create_listTest.expected_image(os.path.join(data_dir, '..','Test_image')))
print(train_inputs)
print(train_expected_outputs)
print(train_expected_outputs)
if train_inputs is not None and train_expected_outputs is not None and test_inputs is not None and test_expected_outputs is not None:
            # Entraînement du MLP sur les données d'entraînement
            num_iterations = 1000
            learning_rate = 0.0004
            print(f"Entraînement du MLP sur {num_iterations} itérations avec un taux d'apprentissage de {learning_rate}")
            for i in range(num_iterations):
                mlp_dll.train(mlp_ptr,
                               train_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                               train_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                               train_inputs.shape[0], train_inputs.shape[1], train_expected_outputs.shape[1],
                               False, 0, learning_rate)
                if i % 100 == 0:
                    # Calcul de l'accuracy sur les données de test
                    correct_predictions = 0
                    total_loss = 0
                    for j in range(test_inputs.shape[0]):
                        input = test_inputs[j]
                        expected_output = test_expected_outputs[j]
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
                        total_loss += np.sum(np.square(expected_output - output))
                    accuracy = correct_predictions / test_inputs.shape[0]
                    average_loss = total_loss / test_inputs.shape[0]
                    print(f"Iteration {i}: Accuracy: {accuracy:.3f}, Loss: {average_loss:.3f}")
            print("Entraînement terminé")
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
                print(f"Expected output: {expected_output}\nPredicted output: {output}\nExpected class: {expected_class}\nPredicted class: {predicted_class}\n")


            # Affichage de la performance du MLP sur les données de test
            accuracy = correct_predictions / test_inputs.shape[0]
            print(f"Nombre de prédictions correctes : {correct_predictions}")
            print(f"Nombre de prédictions incorrectes : {test_inputs.shape[0] - correct_predictions}")
            print(f"Accuracy : {accuracy}")
            # Suppression du MLP
            mlp_dll.deleteMLP(mlp_ptr)