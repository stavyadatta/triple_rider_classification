from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import random
import cv2
import os

datagen = ImageDataGenerator()

# creating X and y labels
imagePaths = sorted(list(paths.list_images("Dataset")))

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = img_to_array(image)

X, y = ...

for imagePath in imagePaths:
    


random.shuffle(imagePaths)

it = datagen.flow(X, y)

image = cv2.imread(imagePath)

