# type: ignore
import os

# Directorios

BASE_DIRECTORY = os.getcwd()
IMAGES_DIRECTORY = os.path.join(BASE_DIRECTORY, 'images')
MODELS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'models')

# Otras Variables
IMG_SHAPE = (256, 256, 3)
NUM_EPOCHS = 5

# Nombre del experimento
EXP_NAME = "test"  # Se puede cambiar dinámicamente si es necesario