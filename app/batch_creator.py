# type: ignore
import os
import sys
import shutil
import random

# ***
import tkinter as tk
from tkinter.filedialog import *
tk.Tk().withdraw()

home_dir = os.path.expanduser("~")

images_set = askdirectory(
    initialdir=home_dir,
    title="Select an images set"    
)
if images_set:
    print("Directory selected: " + images_set)
else:
    print("No directory selected.")
    sys.exit()
# ***

file_name = input("Enter name for new set: ")
if file_name:
    output_dir = f"images\{file_name}"
else:
    print("No name selected.")
    sys.exit()

set_size = input("Enter size for classes set (empty for automatic size): ")
random.seed(42)

def main():
    # Min length images -> Same for every class
    class_images = {cls: os.listdir(os.path.join(images_set, cls)) for cls in os.listdir(images_set) if os.path.isdir(os.path.join(images_set, cls))}
    min_count = min(len(imgs) for imgs in class_images.values())

    # Use min_count if set_size is empty
    set_size_value = int(set_size) if set_size.isdigit() else min_count

    os.makedirs(output_dir, exist_ok=True) # images/

    for cls, imgs in class_images.items():
        class_output_path = os.path.join(output_dir, cls)
        os.makedirs(class_output_path, exist_ok=True)

        selected_imgs = random.sample(imgs, min(set_size_value, len(imgs)))

        for img in selected_imgs:
            shutil.copy(os.path.join(images_set, cls, img), os.path.join(class_output_path, img))

    print(f"New balanced set created at {output_dir} with {set_size_value} images/class")

if __name__ == "__main__":
    main()    