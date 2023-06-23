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


# Chargement des données d'entraînement et de test
# Définition de la structure du MLP
npl = np.array([1875,1024,256,128,3], dtype=np.int64)

mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), npl.size)

data_dir = os.path.join(current_dir, '..', 'Test_image')
train_inputs = np.array(create_listTest.allcolors(os.path.join(data_dir, '..', 'Train_image')))
train_expected_outputs = np.array(create_listTest.expected_image(os.path.join(data_dir, '..', 'Train_image')))
test_inputs = np.array(create_listTest.allcolors(os.path.join(data_dir, '..', 'Test_image')))
test_expected_outputs = np.array(create_listTest.expected_image(os.path.join(data_dir, '..', 'Test_image')))

# Entraînement du MLP sur un nombre spécifique d'époques
samples_inputs = train_inputs
samples_expected_outputs = train_expected_outputs
num_epochs = 75000
learning_rate = 0.001
mlp_dll.train(mlp_ptr,
               samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_inputs.shape[0], samples_inputs.shape[1], samples_expected_outputs.shape[1],
               True, num_epochs, learning_rate)


# Prédire les étiquettes des données de test
predictions = []
for i in range(test_inputs.shape[0]):
    input_data = np.ascontiguousarray(test_inputs[i]).astype(np.float64)
    output = np.ascontiguousarray(np.zeros(3)).astype(np.float64)
    mlp_dll.predict(mlp_ptr, 
                     input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                     input_data.size, 
                     False, 
                     output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                     output.size)
    predicted_class = np.argmax(output)
    expected_class = np.argmax(test_expected_outputs[i])
    predictions.append(predicted_class)
    print(f"Exemple {i+1}: Attendu={expected_class}, Prédit={predicted_class} ({output})")

# Évaluer les prédictions
accuracy = np.mean(np.array(predictions) == np.argmax(test_expected_outputs, axis=1))
print(f"Précision: {accuracy}")



# Suppression du MLP
mlp_dll.deleteMLP(mlp_ptr)