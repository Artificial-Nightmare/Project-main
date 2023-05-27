import ctypes
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'perceptron_multi_couche.dll')
sys.path.append(dll_path)

# Load the DLL
mlp_dll = ctypes.CDLL(dll_path)

# Define the data types for the functions in the DLL
mlp_dll.train.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_double]
mlp_dll.predict.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_double), ctypes.c_int]

# Define the input and output data
samples_inputs = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
samples_expected_outputs = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
outputs = np.zeros((4,), dtype=np.float64)

# Train the MLP
samples_size = 4
inputs_size = 2
outputs_size = 1
iteration_count = 10000
alpha = 0.1
is_classification = True
mlp_dll.train(samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), samples_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), samples_size, inputs_size, outputs_size, is_classification, iteration_count, alpha)

# Predict the outputs
for i in range(4):
    mlp_dll.predict(inputs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)), inputs_size, is_classification, outputs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)), outputs_size)

# Print the outputs
print("Inputs: {}".format(inputs))
print("Outputs: {}".format(outputs))
