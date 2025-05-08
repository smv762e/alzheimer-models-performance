# type: ignore
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import InceptionV3, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, VGG16, VGG19, Xception
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Import configurations
from config import IMG_SHAPE

# Available models
MODEL_DICT = {
    "Inception": InceptionV3,
    "ResNet50": ResNet50,
    "ResNet50V2": ResNet50V2,
    "ResNet101": ResNet101,
    "ResNet101V2": ResNet101V2,
    "ResNet152": ResNet152,
    "ResNet152V2": ResNet152V2,
    "VGG16": VGG16,
    "VGG19": VGG19,
    "Xception": Xception
}

# TRAINING
def select_model_by_name(name):
    return MODEL_DICT[name], name

def build_model(model_fn, num_classes):
    if model_fn in (VGG16, VGG19):
        autoPooling = 'max'
        print("Pooling: max")
    else:
        autoPooling = 'avg'
        print("Pooling: avg")

    base_model = model_fn(include_top=False, weights="imagenet", input_shape=IMG_SHAPE, pooling=autoPooling)
    x = base_model.output
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_callbacks(model_name, model_directory):
    checkpoint_filepath = os.path.join(model_directory, f'{model_name}.keras')

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True)
    
    training_stop = EarlyStopping(monitor='val_loss',
                                  verbose=1, patience=10,
                                  restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=5,
                                  verbose=1,
                                  mode='min',
                                  min_lr=1e-5)
    
    return [model_checkpoint_callback, training_stop, reduce_lr]

def plot_training_history(history, model_name, plot_save_directory):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.tight_layout()
    image_path = os.path.join(plot_save_directory, f'{model_name}_training_history.png')
    plt.savefig(image_path)
    plt.close()
    
    return image_path

# TEST
def confusion(test, y_test, pred2, model_name, plot_save_directory):
    cm = confusion_matrix(y_test, pred2)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues')
    ticks = np.arange(0.5, len(test.class_indices) + 0.5, 1)
    plt.xticks(ticks, rotation=25, labels=test.class_indices)
    plt.yticks(ticks, rotation=0, labels=test.class_indices)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix: " + model_name)
    
    image_path = os.path.join(plot_save_directory, f'{model_name}_confusion.png')
    plt.savefig(image_path)
    plt.close()

    return image_path