import os
import math
import shutil
from tkinter import Tk, Label, Button, Frame, filedialog
from PIL import Image, ImageTk
import numpy as np
import resize
import trainer
import ctypes
import sys

root = Tk()
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
data_dir = os.path.join(current_dir, "Application", "imageAtest")
destination_folder = os.path.join(os.getcwd(), "Application", "imageAtest")

# Supprimer les fichiers existants dans le dossier "imageAtest"
for the_file in os.listdir(destination_folder):
    file_path = os.path.join(destination_folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)
print(data_dir)

def save_image(file_path):
    destination_folder = os.path.join(os.getcwd(), "Application", "imageAtest")
    # Supprimer les fichiers existants dans le dossier "imageAtest"
    for the_file in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    if destination_folder:
        # Vérifier si le dossier existe et le créer s'il n'existe pas
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        # Supprimer les fichiers existants dans le dossier
        for the_file in os.listdir(destination_folder):
            file_path = os.path.join(destination_folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        # Renommer et enregistrer la nouvelle image
        new_file_path = os.path.join(destination_folder, 'image.jpg')
        shutil.copy(file_path, new_file_path)
        print("Image téléchargée avec succès!")

        # Reste du code...

        new_size = (25, 25)
        resize.resize_image(new_file_path,new_file_path, new_size)
        pixels = test_inputs = np.array(allcolors('./Application/imageAtest/'))
        cls = ""
        print(pixels)
        predicted_outputs = trainer.predict(loaded_mlp_ptr, pixels)
        predicted_classes = np.argmax(predicted_outputs, axis=1)
        print("Prédictions du MLP :")
        for i in range(len(predicted_outputs)): 
            predicted_class = predicted_classes[i]
            print(f"Exemple {i+1} - Prédiction : {predicted_class}")
            print(f"Sortie du MLP : {predicted_outputs[i]}")
            if predicted_class == 0:
                cls = "Ballon de baseball"
            elif predicted_class == 1:
                cls = "Ballon de football"
            elif predicted_class == 2:      
                cls = "Ballon de basket"
        print(cls)
        for the_file in os.listdir(destination_folder):
            file_path = os.path.join(destination_folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
print(data_dir)

def allcolors(directory):
    all_pixels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                print(f"L'image {filename} n'est pas en format RGB (mode {image.mode})")
                image = image.convert('RGB')
            pixels = []
            for pixel in image.getdata():
                r, g, b = pixel
                pixels.append((r, g, b))
            all_pixels.append(pixels)
    if all_pixels:
        print("Liste globale de couleurs de pixels :")
        return np.array(all_pixels)
    else:
        print(f"Aucune image valide trouvée dans le dossier {directory}")

def show_selected_image(file_path, default_label, img_label, choose_button):
    with Image.open(file_path) as img:
        # Calculer la nouvelle taille de l'image à 70% de la largeur et de la hauteur
        window_width = root.winfo_width()
        window_height = root.winfo_height()
        img_width = int(0.7 * window_width)
        img_height = int(0.7 * window_height)
        img = img.resize((img_width, img_height), Image.ANTIALIAS)
        choose_button.destroy()
        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo, borderwidth=5, relief="groove")
        img_label.image = photo
        default_label.pack_forget()
        # Supprimer le bouton "Sélectionner une image"
        # Créer un nouveau cadre pour les boutons
        button_frame = Frame(root, bg='#FFFFFF')
        # Ajouter les boutons vert et rouge à droite de l'image
        green_button = Button(button_frame, text="Télécharger", bg="#4CAF50", fg="white",  width=10, command=lambda: save_image(file_path))
        green_button.pack(side="right", padx=100, pady=10)
        red_button = Button(button_frame, text="Autre Image", bg="#EF5350", fg="white", width=7, padx=10,  command=lambda: choose_image(default_label, img_label, choose_button, green_button, red_button, button_frame))
        red_button.pack(side="right", padx=100, pady=10)
        # Ajouter le cadre sous l'image au centre
        button_frame.pack(side="bottom", pady=20)

def choose_image(default_label, img_label, choose_button, green_button, red_button, button_frame):
    green_button.destroy()
    red_button.destroy()
    button_frame.destroy()
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if file_path:
        show_selected_image(file_path, default_label, img_label, choose_button)
        print("Vous avez sélectionné l'image", file_path)

default_text = "Aucune image sélectionnée"
default_label = Label(root, text=default_text, font=("Arial", 24), pady=200, padx=150)
default_label.pack()

# Créer un cadre pour stocker le bouton
button_frame = Frame(root, bg='#FFFFFF')
img_label = Label(root, background='#FFFFFF')
img_label.pack(pady=15)
green_button = Button(button_frame, text="Télécharger", bg="#4CAF50", fg="white",  width=10, command=lambda: save_image(file_path))
green_button.pack(side="right", padx=100, pady=10)
red_button = Button(button_frame, text="Autre Image", bg="#EF5350", fg="white", width=7, padx=10,  command=lambda: choose_image(default_label, img_label, choose_button, green_button, red_button))
red_button.pack(side="right", padx=100, pady=10)
choose_button = Button(button_frame, text="Sélectionner une image", command=lambda: choose_image(default_label, img_label, choose_button, green_button, red_button,button_frame), bg="#4C4C4C", fg="white")
choose_button.pack(pady=15)
# Ajouter le cadre au-dessus du label
button_frame.pack()
root.mainloop()
