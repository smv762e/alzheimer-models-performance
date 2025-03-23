# type: ignore
import os
import sys
import shutil
import random
import tkinter as tk
from tkinter.filedialog import askdirectory

# Ocultar la ventana principal de Tkinter
tk.Tk().withdraw()

# Obtener directorio base del usuario
home_dir = os.path.expanduser("~")

# Seleccionar el conjunto de imágenes
images_set = askdirectory(initialdir=home_dir, title="Select an images set")
if not images_set:
    print("❌ No directory selected. Exiting...")
    sys.exit()

print(f"📂 Directory selected: {images_set}")

# Solicitar nombre del nuevo conjunto
file_name = input("Enter name for new set: ").strip()
if not file_name:
    print("❌ No name selected. Exiting...")
    sys.exit()

# Asegurar que el nombre no contenga caracteres inválidos
invalid_chars = r'\/:*?"<>|'
if any(c in file_name for c in invalid_chars):
    print("❌ Invalid name. Avoid using special characters: / \\ : * ? \" < > |")
    sys.exit()

# Definir directorio de salida de forma segura
output_dir = os.path.join("images", file_name)

# Solicitar el tamaño del conjunto por clase
set_size = input("Enter size for classes set (empty for automatic size): ").strip()
random.seed(42)  # Para reproducibilidad

def main():
    # Obtener imágenes por clase
    class_images = {
        cls: os.listdir(os.path.join(images_set, cls))
        for cls in os.listdir(images_set)
        if os.path.isdir(os.path.join(images_set, cls))
    }

    if not class_images:
        print("❌ No valid class directories found. Exiting...")
        sys.exit()

    # Determinar el número mínimo de imágenes por clase
    min_count = min(len(imgs) for imgs in class_images.values() if imgs)

    if min_count == 0:
        print("❌ Some class folders are empty. Exiting...")
        sys.exit()

    # Determinar el tamaño final del conjunto
    try:
        set_size_value = int(set_size) if set_size.isdigit() else min_count
    except ValueError:
        print("❌ Invalid input for set size. Exiting...")
        sys.exit()

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Copiar imágenes
    for cls, imgs in class_images.items():
        class_output_path = os.path.join(output_dir, cls)
        os.makedirs(class_output_path, exist_ok=True)

        selected_imgs = random.sample(imgs, min(set_size_value, len(imgs)))

        for img in selected_imgs:
            src = os.path.join(images_set, cls, img)
            dst = os.path.join(class_output_path, img)

            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"⚠️ Error copying {img}: {e}")

    print(f"✅ New balanced set created at {output_dir} with {set_size_value} images/class")

if __name__ == "__main__":
    main()
