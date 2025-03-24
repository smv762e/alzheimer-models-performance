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
from config import EXP_NAME

# Modelos disponibles
MODEL_OPTIONS = {
    0: "InceptionV3",  1: "ResNet50",   2: "ResNet50V2",  3: "ResNet101",
    4: "ResNet101V2",  5: "ResNet152",  6: "ResNet152V2",
    7: "VGG16",        8: "VGG19",      9: "Xception"
}

# Función para solicitar un modelo válido
def select_model_input():
    while True:
        try:
            num = int(input("Enter an integer (0-9) to select a model: "))
            if num in MODEL_OPTIONS:
                print(f"✅ Selected Model: {MODEL_OPTIONS[num]}")
                return num
            else:
                print("❌ Invalid selection. Please enter a number between 0 and 9.")
        except ValueError:
            print("❌ Invalid input. Please enter an integer.")
            
# TRAINING
def select_model(num): 
    model_list = [InceptionV3, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, VGG16, VGG19, Xception]
    mod = model_list[num]  
    model_name = mod.__name__
    return mod, model_name

def build_model(num, input_shape, num_classes):
    base_model, _ = select_model(num)

    if num in (0, 1, 2, 3, 4, 5, 6, 9): # normal config
        autoPooling = 'avg'
    if num in (7, 8): # VGG config
        autoPooling = 'max'

    base_model = base_model(include_top=False, weights="imagenet", input_shape=input_shape, pooling=autoPooling)
    x = base_model.output
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_callbacks(model_name, model_directory):
    checkpoint_filepath = os.path.join(model_directory, f'{model_name}.{EXP_NAME}.keras')
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy',
                                                verbose=1, mode='max', save_best_only=True)
    training_stop = EarlyStopping(monitor='loss', verbose=1, patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=0.0001)
    return [model_checkpoint_callback, training_stop, reduce_lr]

def plot_training_history(history, model_name, plot_save_directory, exp_name):
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
    plt.savefig(os.path.join(plot_save_directory, f'{model_name}_{exp_name}_training_history.png'))

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
    plt.savefig(os.path.join(plot_save_directory, f'{model_name}_confusion.png'))