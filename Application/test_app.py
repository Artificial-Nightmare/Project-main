from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import math
import shutil

root = Tk()

# Détermination de la résolution de l'écran
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calcul de la position de la fenêtre au centre de l'écran
window_width = 600
window_height = 600
x_position = math.ceil((screen_width - window_width) / 2)
y_position = math.ceil((screen_height - window_height) / 2)

root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
root.resizable(False, False)
root.title("Analyse de ballons")
root.configure(background='#FFFFFF')


# Sélectionner le répertoire parent de votre fichier Python
file_dir = os.path.dirname(os.path.abspath(__file__))

icon_path = os.path.join(file_dir, "icon.ico")

# Charger l'icône
if os.path.isfile(icon_path):
    icon_image = Image.open(icon_path)
    icon_photo = ImageTk.PhotoImage(icon_image)
    root.iconphoto(True, icon_photo)
else:
    print(f"Attention : impossible de trouver {icon_path}.")

def save_image(file_path):
    destination_folder = '/Application/imageAtest/'

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
    shutil.move(file_path, new_file_path)

def remove_image(default_label, img_label, choose_button, button_frame):
    # Supprimer l'image
    img_label.config(image='')
    default_label.pack()

    # Supprimer le fichier correspondant
    image_path = os.path.join('./Application/imageAtest/', 'my_image.jpg')
    if os.path.exists(image_path):
        os.remove(image_path)

    # Supprimer les boutons de la frame et remettre le bouton "Sélectionner une image"
    button_frame.destroy()
    choose_button.pack(pady=10)

    # Fermer et rouvrir la fenêtre pour réinitialiser l'application
    root.after(100, lambda: root.destroy())
    root.after(200, lambda: root.mainloop())

def show_selected_image(file_path, default_label, img_label, choose_button):
    with Image.open(file_path) as img:
        # Calculer la nouvelle taille de l'image à 70% de la largeur et de la hauteur
        window_width = root.winfo_width()
        window_height = root.winfo_height()
        img_width = int(0.7 * window_width)
        img_height = int(0.7 * window_height)
        img = img.resize((img_width, img_height), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo, borderwidth=5, relief="groove")
        img_label.image = photo
        default_label.pack_forget()

        # Supprimer le bouton "Sélectionner une image"
        choose_button.destroy()

        # Créer un nouveau cadre pour les boutons
        button_frame = Frame(root, bg='#FFFFFF')

        # Ajouter les boutons vert et rouge à droite de l'image
        green_button = Button(button_frame, text="Télécharger", bg="#4CAF50", fg="white",  width=10, command=lambda: save_image(file_path))
        green_button.pack(side=RIGHT, padx=100, pady=10)

        red_button = Button(button_frame, text="Supprimer", bg="#EF5350", fg="white", width=7, padx=10, command=lambda: remove_image(default_label, img_label))
        red_button.pack(side=RIGHT, padx=100, pady=10)

        # Ajouter le cadre sous l'image au centre
        button_frame.pack(side=BOTTOM, pady=20)

def choose_image(default_label, img_label, choose_button):
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if file_path:
        show_selected_image(file_path, default_label, img_label, choose_button)
        print("Vous avez sélectionné l'image", file_path)

default_text = "Aucune image sélectionnée"
default_label = Label(root, text=default_text, font=("Arial", 24), pady=50, padx=200)
default_label.pack()

# Créer un cadre pour stocker le bouton
button_frame = Frame(root, bg='#FFFFFF')

img_label = Label(root, background='#FFFFFF')
img_label.pack(pady=15)

choose_button = Button(button_frame, text="Sélectionner une image", command=lambda: choose_image(default_label, img_label, choose_button), bg="#4C4C4C", fg="white")
choose_button.pack(pady=15)

# Ajouter le cadre au-dessus du label
button_frame.pack()

root.mainloop()