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
mlp_dll.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(
    ctypes.c_double), ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
mlp_dll.train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(
    ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_double]

# Création du MLP avec 1 entrée, 1 neurone caché et 1 sortie
npl = np.array([1, 1], dtype=np.int32)
mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(
    ctypes.POINTER(ctypes.c_int)), npl.size)

samples_inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.double)
samples_expected_outputs = np.array([2, 1, -2, -1], dtype=np.double).flatten()

mlp_dll.train(mlp_ptr,
              samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
              samples_expected_outputs.ctypes.data_as(
                  ctypes.POINTER(ctypes.c_double)),
              samples_inputs.shape[0], samples_inputs.shape[1], 1,  # Updated argument here
              False, 400000, 0.01)

# Create an array to store the predicted outputs
predicted_outputs = np.zeros(samples_inputs.shape[0], dtype=np.double)

# Call the predict function to get the predicted outputs
mlp_dll.predict(mlp_ptr,
                samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                samples_inputs.shape[0],
                False,
                predicted_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                predicted_outputs.shape[0])

# Print the predicted outputs
print("Predicted Outputs:")
for output in predicted_outputs:
    print(output)
