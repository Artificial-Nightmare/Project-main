import tkinter as tk
import os
from PIL import Image
from tkinter import filedialog

# Fonction appelée lors du clic sur le bouton "Télécharger"
def download_image():
    # Ouverture de la boîte de dialogue pour choisir le fichier à télécharger
    file_path = filedialog.askopenfilename(
        title="Choisir une image",
        filetypes=[
            ("Fichiers PNG", "*.png"),
            ("Fichiers JPG", "*.jpg"),
            ("Tous les fichiers", "*.*"),
        ],
    )
    # Si un fichier a été choisi
    if file_path:
        # Ouverture du fichier avec PIL pour vérifier qu'il s'agit d'une image valide
        try:
            with Image.open(file_path) as img:
                # Vérification que le dossier "imageAtest" existe. Si non, on le crée.
                dirname = os.path.abspath(os.path.dirname(__file__))
                target_dir = os.path.join(dirname, "imageAtest")
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                # Enregistrement de l'image dans le dossier "imageAtest"
                target_path = os.path.join(target_dir, "image.png")
                if os.path.exists(target_path):
                    os.remove(target_path)
                img.save(target_path)
                print(f"L'image {img.filename} a été téléchargée avec succès dans le dossier imageAtest !")
        except:
             print("Le fichier choisi n'est pas une image valide.")

# Création de la fenêtre principale
root = tk.Tk()
root.title("Télécharger une image")
# Configuration de la géométrie de la fenêtre
width = 600
height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - width) // 2
y = (screen_height - height) // 2
root.geometry(f"{width}x{height}+{x}+{y}")
# Création du bouton "Télécharger"
download_button = tk.Button(root, text="Télécharger une image", command=download_image)
download_button.pack(side="top", padx=10, pady=10)

# Affichage de la fenêtre
root.mainloop()