# type: ignore
import os
import shutil
import random
import gradio as gr

def batch_creator_func(images_set, set_name, set_size):
    # Validaciones básicas
    if not images_set or not os.path.isdir(images_set):
        raise gr.Error("❌ Directorio no válido")

    if not set_name:
        raise gr.Error("❌ Nombre del conjunto no puede estar vacío")

    # Validar caracteres inválidos en nombre
    invalid_chars = r'\/:*?"<>|'
    if any(c in set_name for c in invalid_chars):
        raise gr.Error("❌ Nombre inválido. Evita usar / \\ : * ? \" < > |")

    random.seed(42)
    class_images = {
        cls: os.listdir(os.path.join(images_set, cls))
        for cls in os.listdir(images_set)
        if os.path.isdir(os.path.join(images_set, cls))
    }

    if not class_images:
        raise gr.Error("❌ No se encontraron carpetas de clases")

    min_count = min(len(imgs) for imgs in class_images.values() if imgs)
    if min_count == 0:
        raise gr.Error("❌ Alguna carpeta está vacía")

    set_size_value = int(set_size) if set_size and int(set_size) > 0 else min_count

    # Crear carpeta destino
    output_dir = os.path.join("images", set_name)
    os.makedirs(output_dir, exist_ok=True)

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
                raise gr.Error("❌ Error copying {img}: {e}")

    clases = ", ".join(class_images.keys())

    return (f"""✅ Conjunto creado exitosamente  

    📁 Ruta: {output_dir}  
    🖼️ Tamaño por clase: {set_size_value}  
    🧠 Clases: {clases}""")