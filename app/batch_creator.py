# type: ignore
import os
import shutil
import random
import gradio as gr

# Import configurations
from config import IMAGES_DIRECTORY

# Import custom modules
from src.data_utils import split_data, save_images, create_dataframes

def batch_creator_func(images_set, set_name, set_size):
    final_msg = "⚠️ An error occurred during set creation."

    # Basic validations
    if not images_set or not os.path.isdir(images_set):
        raise gr.Error("❌ Invalid directory selected.")

    if not set_name:
        raise gr.Error("❌ Dataset name cannot be empty.")

    # Validate invalid characters in the name
    invalid_chars = r'\/:*?"<>|'
    if any(c in set_name for c in invalid_chars):
        raise gr.Error("❌ Invalid name. Avoid using / \\ : * ? \" < > |")

    random.seed(42)
    class_images = {
        cls: os.listdir(os.path.join(images_set, cls))
        for cls in os.listdir(images_set)
        if os.path.isdir(os.path.join(images_set, cls))
    }

    if not class_images:
        raise gr.Error("❌ No class folders found.")

    min_count = min(len(imgs) for imgs in class_images.values() if imgs)
    if min_count == 0:
        raise gr.Error("❌ Some class folder is empty.")

    set_size_value = int(set_size) if set_size and int(set_size) > 0 else min_count

    # Create base output directory
    output_base_dir = os.path.join(IMAGES_DIRECTORY, set_name)
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    # Create dataframe from original dataset
    full_df = create_dataframes(images_set)

    # Now sample per class
    data_df = full_df.groupby('classes').apply(
        lambda x: x.sample(min(set_size_value, len(x)), random_state=42)
    ).reset_index(drop=True)

    train_df, val_df, test_df = split_data(data_df)

    train_df.to_csv(os.path.join(output_base_dir, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(output_base_dir, 'val_df.csv'), index=False)
    test_df.to_csv(os.path.join(output_base_dir, 'test_df.csv'), index=False)

    # save_images(train_df, output_base_dir, "Train")
    # save_images(val_df, output_base_dir, "Val")
    # save_images(test_df, output_base_dir, "Test")

    classes = ", ".join(sorted(class_images.keys()))

    final_msg = (
        f"✅ Dataset created successfully.\n"
        f"📁 Path: {output_base_dir}\n"
        f"🖼️ Size per class: {set_size_value}\n"
        f"🧠 Classes: {classes}\n"
        f"🔹 Train size: {len(train_df)} images\n"
        f"🔹 Val size: {len(val_df)} images\n"
        f"🔹 Test size: {len(test_df)} images"
    )

    return final_msg