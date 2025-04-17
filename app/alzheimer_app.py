import gradio as gr

# Import configurations
from config import IMAGES_DIRECTORY, MODELS_DIRECTORY

# Import custom functions
from batch_creator import batch_creator_func
from train_val import train_val_func
from test_eval import test_eval_func
from multi_test import multi_test_func

info_tab = gr.Interface(
    fn=lambda: "🧠 Welcome to the Alzheimer prediction app.\n\nUse the tabs to create datasets, train models, and make predictions.",
    inputs=[],
    outputs=gr.Markdown(),
    description="# General Information",
    flagging_mode="never",
    clear_btn=None
)

batch_creator = gr.Interface(
    fn=batch_creator_func,
    inputs=[gr.FileExplorer(label="Select image folder", file_count="single", root_dir=IMAGES_DIRECTORY, ignore_glob=("*.txt")),
            gr.Textbox(label="Name of the new dataset"),
            gr.Number(label="Number of images per class", minimum=0, key=int)],
    outputs=gr.TextArea(label="Results"),
    description="# Image Dataset Creator",
    flagging_mode="never"
)

train_val = gr.Interface(
    fn=train_val_func,
    inputs=[gr.FileExplorer(label="Select image folder", file_count="single", root_dir=IMAGES_DIRECTORY, ignore_glob=("*.txt")),
            gr.Radio(["Inception", "ResNet50", "ResNet50V2",
                      "ResNet101", "ResNet101V2", "ResNet152",
                      "ResNet152V2", "VGG16", "VGG19", "Xception"], label="Select a model to train"),
            gr.Number(label="Number of training iterations", minimum=0, key=int)],
    outputs=[gr.TextArea(label="Results"),
             gr.Image(label="Training Evolution", show_download_button=False)],
    description="# Model Training and Validation",
    flagging_mode="never"
)

test_val = gr.Interface(
    fn=test_eval_func,
    inputs=gr.Image(sources="upload", type="pil"),
    outputs=gr.JSON(),
    description="Model Testing",
    flagging_mode="never"
)

multi_test = gr.Interface(
    fn=multi_test_func,
    inputs=gr.Image(sources="upload", type="pil"),
    outputs=gr.JSON(),
    description="Single Image Prediction",
    flagging_mode="never"
)

app = gr.TabbedInterface(
    [info_tab, batch_creator, train_val, test_val, multi_test],
    ["Introduction", "Batch Creator", "Training and Validation", "Model Test", "Single Predictions"],
    title="🧠 Alzheimer Prediction App"
)

if __name__ == "__main__":
    app.launch(inbrowser=True)