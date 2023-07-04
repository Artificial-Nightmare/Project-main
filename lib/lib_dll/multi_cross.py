import ctypes
import numpy as np
import os
import sys

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

# Création du MLP avec 2 entrées, 2 neurones cachés et 1 sortie
npl = np.array([2,64,64, 3], dtype=np.int32)
mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), npl.size)

X = np.random.random((1000, 2)) * 2.0 - 1.0
samples_inputs = np.array(X, dtype=np.double)
Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in X])
samples_expected_outputs = np.array(Y, dtype=np.double)
mlp_dll.train(mlp_ptr,
               samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_inputs.shape[0], samples_inputs.shape[1], 1,
               True, 500000, 0.000012)

correct_predictions = 0
total_predictions = samples_inputs.shape[0]
# Utilisation de la fonction "predict" pour prédire les sorties pour chaque entrée
for i in range(samples_inputs.shape[0]):
    input = samples_inputs[i]
    output = np.zeros(samples_expected_outputs.shape[1], dtype=np.double)

    mlp_dll.predict(mlp_ptr, 
                    input.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                    input.size, 
                    True, 
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                    output.size)

    predicted_label = np.argmax(output)
    expected_label = np.argmax(samples_expected_outputs[i])

    print("Entrée :", input)
    print("Sortie prédite :", output)
    print("Sortie attendue :", samples_expected_outputs[i])

    if predicted_label == expected_label:
        correct_predictions += 1

    print()

accuracy = correct_predictions / total_predictions
print("Taux de réussite : {:.2%}".format(accuracy))     

# Suppression du MLP
mlp_dll.deleteMLP(mlp_ptr)
