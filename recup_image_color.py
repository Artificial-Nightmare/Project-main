from PIL import Image
import os

image_Foot_dir = 'dataset/Balle_de_Football/'
image_Basketball_dir = 'dataset/Balle_de_basketball/'
image_Baseball_dir = 'dataset/Balle_de_Football/'
image_prefix = 'Image_'
image_extension = '.jpg'

image_foot_color = [] #on les stock dans des dico
image_Basketball_color = []
image_Baseball_color = []

for i in range(1, 3):  # remplacez 4 par le nombre total d'images à traiter pour le foot
    image_name = image_prefix + str(i) + image_extension
    image_path = os.path.join(image_Foot_dir, image_name)
    image_foot = Image.open(image_path)
    
    r_sum, g_sum, b_sum = 0, 0, 0
    for pixel in image_foot.getdata():
        r_sum += pixel[0]
        g_sum += pixel[1]
        b_sum += pixel[2]
    r_avg = (image_foot.size[0] * image_foot.size[1] / r_sum)
    g_avg = image_foot.size[0] * image_foot.size[1] / g_sum 
    b_avg = image_foot.size[0] * image_foot.size[1] / b_sum 
    
    image_foot_color.append((r_avg, g_avg, b_avg))
print("balle de foot bg =", image_foot_color)

for i in range(1, 3):  # remplacez 4 par le nombre total d'images à traiter pour le basket
    image_name = image_prefix + str(i) + image_extension
    image_path = os.path.join(image_Basketball_dir, image_name)
    image_basketball = Image.open(image_path)
    
    r_sum, g_sum, b_sum = 0, 0, 0
    for pixel in image_basketball.getdata():
        r_sum += pixel[0]
        g_sum += pixel[1]
        b_sum += pixel[2]
    r_avg = (image_basketball.size[0] * image_basketball.size[1] / r_sum)
    g_avg = image_basketball.size[0] * image_basketball.size[1] / g_sum
    b_avg = image_basketball.size[0] * image_basketball.size[1] / b_sum
    
    image_Basketball_color.append((r_avg, g_avg, b_avg))
print("basketball = ",image_Basketball_color)

for i in range(1, 3):  # remplacez 4 par le nombre total d'images à traiter pour le baseball
    image_name = image_prefix + str(i) + image_extension
    image_path = os.path.join(image_Baseball_dir, image_name)
    image_baseball = Image.open(image_path)

    r_sum, g_sum, b_sum = 0, 0, 0
    for pixel in image_baseball.getdata():
        r_sum += pixel[0]
        g_sum += pixel[1]
        b_sum += pixel[2]
    r_avg = (image_baseball.size[0] * image_baseball.size[1] / r_sum)
    g_avg = image_baseball.size[0] * image_baseball.size[1] / g_sum
    b_avg = image_baseball.size[0] * image_baseball.size[1] / b_sum

    image_Baseball_color.append((r_avg, g_avg, b_avg))
print("baseball = ",image_Baseball_color)
