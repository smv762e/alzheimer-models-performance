# type: ignore
import os
import sys
from config import IMAGES_DIRECTORY, MODELS_DIRECTORY, NUM, IMG_SHAPE, NUM_EPOCHS, EXP_NAME
import pandas as pd
from src.data_utils import create_dataframes, split_data, create_image_generators
from src.log_utils import Tee
from src.model_utils import select_model, build_model, create_callbacks, plot_training_history
from datetime import datetime

# ***
import tkinter as tk
from tkinter.filedialog import *
tk.Tk().withdraw()

images_set = askdirectory(
    initialdir=IMAGES_DIRECTORY,
    title="Select an images set"    
)
if images_set:
    print("Directory selected: " + images_set)
else:
    print("No directory selected.")
    sys.exit()
# ***

DATE = datetime.now().strftime("%Y-%m-%d_%H-%M")

def main():
    # Create dataframe/subdirectory
    data_df = create_dataframes(images_set)
    num_classes = data_df['classes'].nunique()
    
    # Split in train, val and test
    train_df, val_df, test_df = split_data(data_df)
    
    train_gen = create_image_generators(train_df)
    val_gen= create_image_generators(val_df)
    test_gen = create_image_generators(test_df)

    # Build model
    model = build_model(NUM, IMG_SHAPE, num_classes)
    
    _, model_name = select_model(NUM)
    model_directory = os.path.join(MODELS_DIRECTORY, model_name, DATE)
    os.makedirs(model_directory, exist_ok=True) # models/model_name/date/

    log_path = os.path.join(model_directory, "training_log.txt")
    with open(log_path, "w") as log_file:
        sys.stdout = Tee(log_file)

        print("\nLabels Distribution:")
        print(data_df["classes"].value_counts())
        model.summary()

        callbacks = create_callbacks(model_name, model_directory)
        
        # Training and validation
        history = model.fit(train_gen, epochs=NUM_EPOCHS, validation_data=val_gen, callbacks=callbacks)
        history_data = history.history
        history_df = pd.DataFrame(history_data)
        history_directory = os.path.join(model_directory, 'training_history.csv')
        history_df.to_csv(history_directory, index=False)

        plot_training_history(history, model_name, model_directory, EXP_NAME)
        
        print("Evaluating model on test data...")
        _, test_acc = model.evaluate(test_gen, verbose=1)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()