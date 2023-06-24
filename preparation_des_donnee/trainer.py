import os 
from PIL import Image
import numpy as np
import ctypes
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'perceptron_multi_couche.dll')
sys.path.append(dll_path)

# Load the DLL
mlp_dll = ctypes.cdll.LoadLibrary(dll_path)

# Définition des types de données
mlp_dll.deleteMLP.argtypes = [ctypes.c_void_p]
mlp_dll.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
mlp_dll.train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_double]


# Train the MLP on a specific number of epochs
def training(num_epochs, learning_rate,samples_inputs,samples_expected_outputs,mlp_ptr):
    mlp_dll.train(mlp_ptr,
               samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               samples_inputs.shape[0], samples_inputs.shape[1], samples_expected_outputs.shape[1],
               True, num_epochs, learning_rate)
    return mlp_ptr
    
def predict(mlp_ptr, inputs):
    # Convertir les tableaux numpy en tableaux de type ctypes
    inputs_ptr = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # Obtenir les tailles des entrées et des sorties
    inputs_size = inputs.shape[1]
    outputs_size = 3  # Remplacez 3 par le nombre de classes dans votre cas
    
    # Créer un tableau numpy pour les sorties
    outputs = np.zeros((inputs.shape[0], outputs_size), dtype=np.double)
    
    # Appeler la fonction predict de la DLL
    mlp_dll.predict(mlp_ptr, inputs_ptr, inputs_size, False,
                    outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), outputs_size)
    
    return outputs






