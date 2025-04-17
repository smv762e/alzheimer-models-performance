# type: ignore
import sys
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter.filedialog import askopenfilename

# Importar configuraciones
from config import MODELS_DIRECTORY, IMG_SHAPE

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

# Función de predicción
def multi_test_func(img):
    model_path = select_model_directory()

    # Cargar modelo
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit()

    # Preprocesar la imagen como lo hacías en entrenamiento
    img = img.resize((IMG_SHAPE[0],IMG_SHAPE[1]))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar si aplicaba

    # Predecir
    preds = model.predict(img_array)[0]
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = float(preds[pred_index])

    return {label: float(p) for label, p in zip(class_names, preds)}

# Cargar clases
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Crear interfaz de Gradio
interface = gr.Interface(
    fn=multi_test_func,
    inputs=gr.Image(sources="upload", type="pil"),
    outputs=gr.JSON(),
    title="🧠 Alzheimer Prediction",
    description="Carga una imagen de resonancia y el modelo predice la etapa."
)

if __name__ == "__main__":
    interface.launch()