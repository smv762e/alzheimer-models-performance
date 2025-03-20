# type: ignore
import os

# 0 - PATHS

BASE_DIRECTORY = os.getcwd()
IMAGES_DIRECTORY = os.path.join(BASE_DIRECTORY, 'images')
MODELS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'models')

# 1 - MODELS

# 0 = InceptionV3,  1 = ResNet50,   2 = ResNet50V2,     3 = ResNet101,
# 4 = ResNet101V2,  5 = ResNet152,  6 = ResNet152V2,
# 7 = VGG16,        8 = VGG19,      9 = Xception

NUM = 9 # Change this index to select a model

# Other Variables
IMG_SHAPE = (256, 256, 3)
NUM_EPOCHS = 5

# Save Model / Load Model
EXP_NAME = 'test'  # Pick a name