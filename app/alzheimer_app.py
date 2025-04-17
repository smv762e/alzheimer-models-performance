import gradio as gr

# Importar configuraciones
from config import IMAGES_DIRECTORY, MODELS_DIRECTORY

# Importar funciones propias
from batch_creator import batch_creator_func
from train_val import train_val_func
from test_eval import test_eval_func
from multi_test import multi_test_func

batch_creator = gr.Interface(
    fn=batch_creator_func,
    inputs=[gr.FileExplorer(label="Selecciona carpeta de imágenes", file_count="single", root_dir=IMAGES_DIRECTORY, ignore_glob=("*.txt")),
            gr.Textbox(label="Nombre del nuevo set"),
            gr.Number(label="Número de imágenes por clase", minimum=0, key=int)],
    outputs=gr.TextArea(label="Resultados"),
    description="# Creador de conjuntos de imágenes",
    flagging_mode="never"
)

train_val = gr.Interface(
    fn=train_val_func,
    inputs=[gr.FileExplorer(label="Selecciona carpeta de imágenes", file_count="single", root_dir=IMAGES_DIRECTORY, ignore_glob=("*.txt")),
            gr.Dropdown(["Inception", "ResNet50", "ResNet50V2",
                               "ResNet101", "ResNet101V2", "ResNet152",
                                 "ResNet152V2", "VGG16", "VGG19", "Xception"],label="Selecciona un modelo para entrenar"),
            gr.Number(label="Número de imágenes por clase", minimum=0, key=int)],
    outputs=gr.JSON(),
    description="Entrenamiento y validación de modelos"
)

test_val = gr.Interface(
    fn=test_eval_func,
    inputs=gr.Image(sources="upload", type="pil"),
    outputs=gr.JSON(),
    description="Prueba de modelos"
)

multi_test = gr.Interface(
    fn=multi_test_func,
    inputs=gr.Image(sources="upload", type="pil"),
    outputs=gr.JSON(),
    description="PRedición de imágenes individuales"
)

app = gr.TabbedInterface(
    [batch_creator, train_val, test_val, multi_test], 
    ["Creador de conjuntos", "Entrenamiento y validación", "Prueba del modelo", "Pruebas individuales"],
    title="🧠 Alzheimer Prediction App")

if __name__ == "__main__":
    app.launch(inbrowser=True)