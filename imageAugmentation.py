from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import sys
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",
	help="path to the input image", default="Dataset/triple_person")
ap.add_argument("-o", "--output",
	help="path to output directory to store augmentation examples", default="augData/triple_person")
ap.add_argument("-t", "--total", type=int, default=10,
	help="# of training samples to generate")
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args["dataset"])))

aug = ImageDataGenerator(
    featurewise_center=True,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
    horizontal_flip=True,
	fill_mode="nearest")


# looping over image directory for image 
for imagePath in imagePaths:    
    img = load_img(imagePath)
    img = img_to_array(img)
    image = np.expand_dims(img, axis = 0)
    
    total = 0
    
    print("[INFO] generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
	save_prefix="image", save_format="jpg")
    
    for image in imageGen:
        total += 1
    
        if total == args["total"]:
            print("entering this")
            break