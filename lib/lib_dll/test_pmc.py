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
mymlp = ctypes.cdll.LoadLibrary(dll_path)

# Define the input and output types of the functions in the DLL

mymlp.MyMLP_train.restype = None
mymlp.MyMLP_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_double]
mymlp.MyMLP_predict.restype = None
mymlp.MyMLP_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
mymlp.MyMLP_destroy.restype = None
mymlp.MyMLP_destroy.argtypes = [ctypes.c_void_p]

# Define the input-output pairs to train the MLP
samples_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.double)
samples_expected_outputs = np.array([[0], [1], [1], [0]], dtype=np.double)
samples_size = samples_inputs.shape[0]
inputs_size = samples_inputs.shape[1]
outputs_size = samples_expected_outputs.shape[1]

# Create an instance of the MyMLP class
npl = [2, 4, 1]
mymlp_instance = mymlp.MyMLP_create(ctypes.c_void_p, ctypes.c_int * len(npl)(*npl))

# Train the MLP
iteration_count = 10000
alpha = 0.1
is_classification = True
mymlp.MyMLP_train(mymlp_instance, samples_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), samples_expected_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), samples_size, inputs_size, outputs_size, is_classification, iteration_count, alpha)

# Test the MLP
for i in range(4):
    inputs = np.array([samples_inputs[i]], dtype=np.double)
    output = np.zeros((1,), dtype=np.double)
    mymlp.MyMLP_predict(mymlp_instance, inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), inputs_size, is_classification, output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), outputs_size)
    print(f"inputs: {inputs.flatten()} -> output: {output[0]}")

# Free the memory used by the MyMLP instance
mymlp.MyMLP_destroy(mymlp_instance)
