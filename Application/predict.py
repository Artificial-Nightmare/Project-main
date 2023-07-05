import pickle
import trainer
import os 
import numpy as np
import create_listTest
import ctypes
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path to the DLL to the system's search path
dll_path = os.path.join(current_dir, 'perceptron_multi_couche.dll')
sys.path.append(dll_path)

# Load the DLL
mlp_dll = ctypes.cdll.LoadLibrary(dll_path)
mlp_dll.saveModel.argtypes = [ctypes.c_void_p]
mlp_dll.loadModel.restype = ctypes.c_void_p


loaded_mlp_ptr = mlp_dll.loadModel()
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'imageAtest')
test_inputs = np.array(create_listTest.allcolors(os.path.join(data_dir, 'imageAtest')))

predicted_outputs = trainer.predict(loaded_mlp_ptr, test_inputs)

# Obtenir l'indice de la classe prédite pour chaque prédiction
# Obtenir l'indice de la classe attendue pour chaque étiquette
predicted_classes = np.argmax(predicted_outputs, axis=1)

# Afficher les classes prédites et les classes attendues côte à côte
print("Prédictions du MLP :")
for i in range(len(predicted_classes)):
    predicted_class = predicted_classes[i]
    print(f"Exemple {i+1} - Prédiction : {predicted_class}")
    print(f"Sortie du MLP : {predicted_outputs[i]}")
