from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import shutil

# Fonction appelée lors du clic sur le bouton "Télécharger"
def download_image():
    global img_label
    global img_path
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
                # Récupération de la taille de l'image
                w, h = img.size
                # Calcul de la taille maximale de l'image pour ne pas dépasser 50% de la taille de la fenêtre
                max_w = root.winfo_width() // 2 if root.winfo_width() else 400
                max_h = root.winfo_height() // 2 if root.winfo_height() else 400
                if max_w < w or max_h < h:
                    ratio = min(max_w / w, max_h / h)
                    target_size = (int(w * ratio), int(h * ratio))
                    img = img.resize(target_size, Image.ANTIALIAS)
                # Conversion de l'image PIL en objet PhotoImage Tkinter
                photo = ImageTk.PhotoImage(img)
                # Affichage de l'image dans un widget Label
                img_label.config(image=photo)
                img_label.image = photo
                img_label.pack(expand=True, fill=BOTH)
                # Stockage du chemin de l'image pour une utilisation ultérieure
                img_path = file_path
        except:
             print("Le fichier choisi n'est pas une image valide.")

# Création de la fenêtre principale
root = Tk()
root.title('Téléchargement d\'images')
# Configuration de la géométrie de la fenêtre
width = 800
height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - width) // 2
y = (screen_height - height) // 2
root.geometry(f"{width}x{height}+{x}+{y}")
# Gestion du redimensionnement de la fenêtre par l'utilisateur
root.resizable(width=True, height=True)

# Création d'un widget Label pour afficher l'image sélectionnée ou un message par défaut
default_image = Image.open("default_image.png") # Remplacer par le chemin d'une image par défaut
default_photo = ImageTk.PhotoImage(default_image)
img_label = Label(root, image=default_photo)
img_label.image = default_photo
img_label.pack(expand=True, fill=BOTH)

# Création du bouton "Télécharger"
download_button = Button(root, text="Télécharger une image", command=download_image)
download_button.pack(side="left", padx=10, pady=10)

# Fonction appelée lors du clic sur le bouton "Valider"
def validate_image():
    global img_path
    if img_path is not None:
        # Si l'utilisateur a choisi une image, enregistrement de l'image dans le dossier
        # Vérification que le dossier existe, création si nécessaire
        target_dir = "imageAtest"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        # Copie de l'image dans le dossier
        target_path = os.path.join(target_dir, "image.png")
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.copy(img_path, target_path)
        print(f"L'image {img_path} a été téléchargée avec succès dans le dossier {target_dir} !")

# Fonction appelée lors du clic sur le bouton "Annuler"
def cancel():
    global img_path
    # Réinitialisation de l'image à l'image par défaut
    img_label.config(image=default_photo)
    img_label.image = default_photo
    # Suppression du chemin de l'image sélectionnée
    img_path = None

# Création des boutons "Valider" et "Annuler"
validate_button = Button(root, text="Valider", fg="green", command=validate_image)
validate_button.pack(side="bottom", padx=10, pady=10)
cancel_button = Button(root, text="Annuler", fg="red", command=cancel)
cancel_button.pack(side="bottom", padx=10, pady=10)

# Variable pour stocker le chemin du fichier image sélectionné
img_path = None

# Lancement de la boucle principale d'événements
root.mainloop()