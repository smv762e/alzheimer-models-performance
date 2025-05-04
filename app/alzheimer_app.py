import gradio as gr

# Import configurations
from config import IMAGES_DIRECTORY, MODELS_DIRECTORY

# Import custom functions
from batch_creator import batch_creator_func
from train_val import train_val_func
from test_eval import test_eval_func
from multi_test import multi_test_func

# Load README.md
def load_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise gr.Error(f"❌ Failed to load README.md: {e}")

with gr.Blocks() as info_tab:
    gr.Markdown(load_readme())

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
    inputs=[gr.FileExplorer(label="Select image training folder", file_count="single", root_dir=IMAGES_DIRECTORY, glob=("*.csv")),
            gr.FileExplorer(label="Select image validation folder", file_count="single", root_dir=IMAGES_DIRECTORY, glob=("*.csv")),
            gr.Radio(["Inception", "ResNet50", "ResNet50V2",
                      "ResNet101", "ResNet101V2", "ResNet152",
                      "ResNet152V2", "VGG16", "VGG19", "Xception"], label="Select a model to train"),
            gr.Number(label="Number of training iterations", minimum=0, key=int)],
    outputs=[gr.TextArea(label="Results"),
             gr.Dataframe(label="Training History", headers=("loss", "accuracy", "val_loss", "val_accuracy", "lr")),
             gr.Image(label="Training Evolution", show_download_button=False)],
    description="# Model Training and Validation",
    flagging_mode="never"
)

test_val = gr.Interface(
    fn=test_eval_func,
    inputs=[gr.FileExplorer(label="Select image testing folder", file_count="single", root_dir=IMAGES_DIRECTORY, glob=("*.csv")),
            gr.FileExplorer(label="Select a trained model", file_count="single", root_dir=MODELS_DIRECTORY, glob=("*.keras"))],
    outputs=[gr.TextArea(label="Results"),
             gr.Image(label="Training Evolution", show_download_button=False)],
    description="# Model Evaluation",
    flagging_mode="never"
)

multi_test = gr.Interface(
    fn=multi_test_func,
    inputs=[gr.Image(label= "Upload an image", sources="upload", type="pil"),
            gr.FileExplorer(label="Select a trained model", file_count="single", root_dir=MODELS_DIRECTORY, glob=("*.keras"))],
    outputs=gr.TextArea(label="Results"),
    description="# Single Image Prediction",
    flagging_mode="never"
)

app = gr.TabbedInterface(
    [info_tab, batch_creator, train_val, test_val, multi_test],
    ["Introduction", "Batch Creator", "Training and Validation", "Model Evaluation", "Single Predictions"],
    title="🧠 Alzheimer Prediction App"
)

if __name__ == "__main__":
    app.launch(inbrowser=True)