# type: ignore
import os
import sys
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askdirectory

# Importar configuraciones
from config import IMAGES_DIRECTORY, MODELS_DIRECTORY, NUM_EPOCHS

# Importar módulos propios
from src.data_utils import create_dataframes, split_data, create_image_generators
from src.log_utils import Tee
from src.model_utils import select_model_input, select_model, build_model, create_callbacks, plot_training_history

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
# 📌 Función principal
# -------------------------------
def main():
    num = select_model_input()
    images_set = select_images_directory()
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

    try:
        # Crear DataFrame con las imágenes
        data_df = create_dataframes(images_set)
        num_classes = data_df['classes'].nunique()

        # Dividir en train, val y test
        train_df, val_df, test_df = split_data(data_df)

        # Crear generadores de imágenes
        train_gen = create_image_generators(train_df)
        val_gen = create_image_generators(val_df)
        test_gen = create_image_generators(test_df)

        # Construir modelo
        model = build_model(num, num_classes)
        _, model_name = select_model(num)

        # Crear directorio para guardar el modelo
        model_directory = os.path.join(MODELS_DIRECTORY, model_name, date_str)
        os.makedirs(model_directory, exist_ok=True)

        # Configurar redirección de salida
        log_path = os.path.join(model_directory, "training_log.txt")
        summary_path = os.path.join(model_directory, "model_summary.txt")

        with open(log_path, "w", encoding="utf-8") as log_file, \
            open(summary_path, "w", encoding="utf-8") as summary_file:

            sys.stdout = Tee(log_file)

            print("\n📊 Labels Distribution:")
            print(data_df["classes"].value_counts())

            # Guardar resumen del modelo en archivo separado
            model.summary(print_fn=lambda x: summary_file.write(x + "\n"))

            # Crear callbacks y entrenar modelo
            callbacks = create_callbacks(model_name, model_directory)
            print("\n💡 Training in progress...")
            history = model.fit(train_gen, epochs=NUM_EPOCHS, validation_data=val_gen, callbacks=callbacks)

            # Guardar historial de entrenamiento
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(os.path.join(model_directory, 'training_history.csv'), index=False)

            # Graficar historial
            plot_training_history(history, model_name, model_directory)

            # Evaluación final del modelo
            print("🔍 Evaluating model on test data...")
            _, test_acc = model.evaluate(test_gen, verbose=1)
            print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

    except Exception as e:
        print(f"❌ Error during execution: {e}")

    finally:
        sys.stdout = sys.__stdout__  # Restaurar la salida estándar

# -------------------------------
# 📌 Punto de entrada del script
# -------------------------------
if __name__ == "__main__":
    main()