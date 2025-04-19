# type: ignore
import os

# DEFAULT SETTINGS (NOT RECOMMENDED TO MODIFY)

# Directories
BASE_DIRECTORY = os.getcwd()
IMAGES_DIRECTORY = os.path.join(BASE_DIRECTORY, 'images')
MODELS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'models')

# Image size
IMG_SHAPE = (256, 256, 3)

# Load class names
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']