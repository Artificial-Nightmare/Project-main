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
npl = np.array([2,3, 3], dtype=np.int32)
mlp_ptr = mlp_dll.createMLP(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), npl.size)

# Entraînement du MLP sur le XOR
X = np.random.random((50, 2)) * 2.0 - 1.0
Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else 
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else 
              [0, 0, 0]for p in X])

X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]
samples_inputs = np.array(X, dtype=np.double)
samples_expected_outputs = np.array(Y, dtype=np.double)
mlp_dll.train(mlp_ptr,
               samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_inputs.shape[0], samples_inputs.shape[1], 1,
               True, 1000000,0.05)


input_size = 2
output_size = samples_expected_outputs.shape[1]

input = np.zeros(input_size, dtype=np.double)
output = np.zeros(output_size, dtype=np.double)

correct_predictions = 0

for i in range(samples_inputs.shape[0]):
    input[0] = samples_inputs[i, 0]
    input[1] = samples_inputs[i, 1]
    mlp_dll.predict(mlp_ptr, 
                    input.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                    input_size, 
                    True, 
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                    output_size)
    expected_class = np.argmax(samples_expected_outputs[i])
    predicted_class = np.argmax(output)
    print(f"[{input[0]}, {input[1]}] Expected Class: {expected_class}, Predicted Class: {predicted_class}")
    
    if expected_class == predicted_class:
        correct_predictions += 1

accuracy = (correct_predictions / samples_inputs.shape[0]) * 100
print(f"\nAccuracy: {accuracy}%")





# Suppression du MLP
mlp_dll.deleteMLP(mlp_ptr)
