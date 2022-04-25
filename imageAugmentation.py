from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import random
import numpy as np

DATASET = "Dataset/single_person"
OUTPUT_PATH = "testSet/single_person"
AUG_PER_IMG = 1

# Generates Images from the dataset and can possibly augment random amount of images from set
def GeneratorFunction(dataset_path, output_path, aug_per_img, random_num=None):
    imagePaths = sorted(list(paths.list_images(dataset_path)))
    
    if random_num:
        imagePaths = random.choices(imagePaths, k=random_num)

    aug = ImageDataGenerator(
        featurewise_center=True,
        width_shift_range=0.2,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest")


    # looping over image directory for image 
    for imagePath in imagePaths:    
        img = load_img(imagePath)
        img = img_to_array(img)
        image = np.expand_dims(img, axis = 0)
        
        total = 0
        
        print("[INFO] generating images...")
        imageGen = aug.flow(image, batch_size=8, save_to_dir=output_path,
        save_prefix="image", save_format="jpg")
        
        for image in imageGen:
            total += 1
        
            if total == aug_per_img:
                print("entering this")
                break