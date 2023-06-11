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
npl = np.array([2, 2, 1], dtype=np.int32)
mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), npl.size)

# Entraînement du MLP sur le XOR
samples_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.double)
samples_expected_outputs = np.array([[0], [1], [1], [0]], dtype=np.double)
mlp_dll.train(mlp_ptr,
               samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_inputs.shape[0], samples_inputs.shape[1], 
               samples_expected_outputs.shape[1], 
               True, 
               200000, 
               0.001)

# Test du MLP sur le XOR
input = np.zeros(2, dtype=np.double)
output = np.zeros(1, dtype=np.double)
for i in range(samples_inputs.shape[0]):
    input[0] = samples_inputs[i, 0]
    input[1] = samples_inputs[i, 1]
    mlp_dll.predict(mlp_ptr, 
                    input.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                    input.size, 
                    True, 
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                    output.size)
    print(f"[{int(input[0])}, {int(input[1])}] = {output[0]}")

# Suppression du MLP
mlp_dll.deleteMLP(mlp_ptr)
