
import os, random
from PIL import Image
from non_gui_networks.lgn_v1 import calc_gaussians
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform, skimage.util



# Pick a random image fragment
def pick_random_image():
    folder = r"/Users/Michael/Documents/SeniorProject/NEXUS/non_gui_networks/lgn_v1/lgn_v1_image_fragments/"

    a = random.choice(os.listdir(folder))
    print(a)
    file = folder + a
    # x = list(Image.open(file).convert('L').getdata())
    return file


# Perform degree of gaussians over a randomly selected image fragment
def degree_of_gaussians():
    # Load random image
    image = pick_random_image()
    # Perform degree of gaussian
    image = calc_gaussians.calc_degree_of_gaussians(image)
    # Put into shape 144 from 12x12
    inverse = skimage.util.invert(image)
    return np.ndarray.flatten(image), np.ndarray.flatten(inverse)

"""
# Extract 50x50 image samples from the 4 600x450 images
# One-time use only; no need to create the image fragments more than once

import cv2

path = "/NEXUS/non_gui_networks/lgn_v1/lgn_v1_input/"
for i in range(4):
    img = cv2.imread(path + "/master_image{}.jpg".format(i+1))
    for r in range(0,img.shape[0],50):
        for c in range(0,img.shape[1],50):
            cv2.imwrite(path+f"img{i}_{r}_{c}.png",img[r:r+50, c:c+50,:])
"""