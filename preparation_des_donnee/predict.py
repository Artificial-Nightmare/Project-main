import pickle
import trainer
import os 
import numpy as np
import create_listTest
import ctypes

with open('saved_model.pkl', 'rb') as file:
    mlp_data = pickle.load(file)
class MLP(ctypes.Structure):
    _fields_ = [
        ('d', ctypes.POINTER(ctypes.c_int)),
        ('L', ctypes.c_int),
        ('W', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))),
        ('X', ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
        ('deltas', ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    ]

mlp = MLP()
mlp.d = ctypes.pointer(ctypes.c_int(mlp_data['d']))
mlp.L = mlp_data['L']
mlp.W = [[(ctypes.c_double * len(row))(*row) for row in layer] for layer in mlp_data['W']]
mlp.X = [(ctypes.c_double * len(row))(*row) for row in mlp_data['X']]
mlp.deltas = [(ctypes.c_double * len(row))(*row) for row in mlp_data['deltas']]

print(mlp_data)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'Test_image')
test_inputs = np.array(create_listTest.allcolors(os.path.join(data_dir, '..', 'Test_image_stockage')))
test_expected_outputs = np.array(create_listTest.expected_image(os.path.join(data_dir, '..', 'Test_image_stockage')))

predicted_outputs = trainer.predict(ctypes.pointer(mlp), test_inputs)

# Obtenir l'indice de la classe prédite pour chaque prédiction
# Obtenir l'indice de la classe attendue pour chaque étiquette
predicted_classes = np.argmax(predicted_outputs, axis=1)
expected_classes = np.argmax(test_expected_outputs, axis=1)

# Afficher les classes prédites et les classes attendues côte à côte
print("Prédictions du MLP :")
for i in range(len(predicted_classes)):
    predicted_class = predicted_classes[i]
    expected_class = expected_classes[i]
    print(f"Exemple {i+1} - Prédiction : {predicted_class}, Attendu : {expected_class}")
    print(f"Sortie du MLP : {predicted_outputs[i]}")
