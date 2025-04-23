# type: ignore
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import configurations
from config import IMAGES_DIRECTORY

# filepaths and labels extractor
def get_filepaths_and_labels(directory):
    filepaths = []
    labels = []
    for subdirectory in os.listdir(directory):
        subpath = os.path.join(directory, subdirectory)
        if os.path.isdir(subpath):
            image_paths = [os.path.join(subpath, img) for img in os.listdir(subpath)]
            filepaths.extend(image_paths)
            labels.extend([subdirectory] * len(image_paths))
    return filepaths, labels

# .../images/X/classes/files -> Dataframe
def create_dataframes(data_path):
    data_filepaths, data_labels = get_filepaths_and_labels(data_path)
    data = {'filepaths': data_filepaths, 'classes': data_labels}
    data_df = pd.DataFrame(data)
    return data_df

# Train = 60%, Val = 20%, Test = 20%
def split_data(data_df, test_size=0.2, val_size=0.25):
    train_data, test_df = train_test_split(data_df, test_size=test_size, random_state=42, stratify=data_df['classes'])
    train_df, val_df = train_test_split(train_data, test_size=val_size, random_state=42, stratify=train_data['classes'])
    return train_df, val_df, test_df

# Save Test for Test_Eval
def save_test_images(test_df):
    output_dir = os.path.join(IMAGES_DIRECTORY, "Test_Split")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in test_df.iterrows():
        src_path = row['filepaths']
        label = row['classes']
        dest_dir = os.path.join(output_dir, label)
        os.makedirs(dest_dir, exist_ok=True)

        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(src_path, dest_path)

# batch_size=16 (limited performance), batch_size=32 (normal)
def create_image_generators(data_df):
    image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
    data = image_gen.flow_from_dataframe(dataframe=data_df, x_col="filepaths", y_col="classes",
                                          target_size=(256, 256), color_mode='rgb',
                                          class_mode="categorical", batch_size=16, shuffle=False)
    return data