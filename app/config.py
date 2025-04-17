# type: ignore
import os

# POR DEFECTO (NO RECOMENDABLE CONFIGURAR)

# Directorios
BASE_DIRECTORY = os.getcwd()
IMAGES_DIRECTORY = os.path.join(BASE_DIRECTORY, 'images')
MODELS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'models')

# Tamaño de las imágenes
IMG_SHAPE = (256, 256, 3)