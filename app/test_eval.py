# type: ignore
import os
import sys
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

# Importar configuraciones
from config import IMAGES_DIRECTORY, MODELS_DIRECTORY

# Importar módulos propios
from src.data_utils import create_dataframes, create_image_generators
from src.log_utils import Tee
from src.model_utils import confusion

# -------------------------------
# 📌 Función para seleccionar directorio de imágenes
# -------------------------------
def select_images_directory():
    tk.Tk().withdraw()  # Ocultar ventana principal de Tkinter
    images_set = askdirectory(initialdir=IMAGES_DIRECTORY, title="Select an images set")
    
    if not images_set:
        print("❌ No directory selected. Exiting...")
        sys.exit()
    
    print(f"📂 Directory selected: {images_set}")
    return images_set

# -------------------------------
# 📌 Función para obtener ruta del modelo entrenado 
# -------------------------------
def select_model_directory():
    tk.Tk().withdraw()  # Ocultar ventana principal de Tkinter
    model_path = askopenfilename(initialdir=MODELS_DIRECTORY, title="Select a .keras model", filetypes=[("Keras Model Files", "*.keras")])
    
    if not model_path:
        print("❌ No model selected. Exiting...")
        sys.exit()
    
    print(f"📄 Model selected: {model_path}")
    return model_path

# -------------------------------
# 📌 Función principal
# -------------------------------
def main():
    model_path = select_model_directory()
    images_set = select_images_directory()
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Crear dataframe
    test_df = create_dataframes(images_set)
    test_gen = create_image_generators(test_df)

    # Cargar modelo
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit()

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_og = os.path.basename(model_path).split('.')[0]
    model_directory = os.path.join(MODELS_DIRECTORY, model_og)
    test_directory = os.path.join(model_directory, "test", date_str)

    # Crear directorio de pruebas
    os.makedirs(test_directory, exist_ok=True)

    log_path = os.path.join(test_directory, "testing_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(log_file)

        print("\n📊 Labels Distribution:")
        print(test_df["classes"].value_counts())

        print("\n🔍 Evaluating model on test data...")
        _, test_acc = model.evaluate(test_gen, verbose=1)
        print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

        # Obtener predicciones
        pred = model.predict(test_gen)
        if pred.ndim > 1:  # Si es una matriz de probabilidades
            pred = np.argmax(pred, axis=1)

        # Convertir índices en etiquetas
        labels = {v: k for k, v in test_gen.class_indices.items()}
        pred_labels = [labels[idx] for idx in pred]

        # Convertir y_test a lista
        y_test = test_df["classes"].tolist()

        # Reporte de clasificación
        print("\n📜 Classification Report:")
        print(classification_report(y_test, pred_labels))

        print("\n🎯 Accuracy of the Model:", "{:.1f}%".format(accuracy_score(y_test, pred_labels) * 100))

        # Generar matriz de confusión
        confusion(test_gen, y_test, pred_labels, model_name, test_directory)

    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
