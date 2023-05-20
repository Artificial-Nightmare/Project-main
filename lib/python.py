import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'liblinear_classification.dll')
sys.path.append(dll_path)

# Load the DLL
dll = ctypes.CDLL(dll_path)

# Define the input and output data types of the function
dll.linear_classification.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
dll.linear_classification.restype = None

# Prepare the data
X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
Y = [1, -1, 1]

# Flatten the input data
X_flat = [val for sublist in X for val in sublist]

# Define the data types of the input arrays
X_array = (ctypes.c_double * len(X_flat))(*X_flat)
Y_array = (ctypes.c_int * len(Y))(*Y)

learning_rate = ctypes.c_double(0.01)
max_iterations = ctypes.c_int(1000)

# Create a buffer for the output data
w = (ctypes.c_double * (len(X[0]) + 1))()

# Call the function
dll.linear_classification(X_array, Y_array, len(X), len(X[0]), learning_rate, max_iterations, w)

# Convert the results to a Python list
w_list = list(w)

# Plot the classification
plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=np.array(Y))
x = np.linspace(0, 10, 100)
y = (-w_list[0] - w_list[1] * x) / w_list[2]
plt.plot(x, y)
plt.show()
