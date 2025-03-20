# type: ignore
import os
import sys
from config import IMAGES_DIRECTORY, MODELS_DIRECTORY
from tensorflow.keras.models import load_model
from src.data_utils import create_dataframes, create_image_generators
from src.log_utils import Tee
from src.model_utils import confusion
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
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
    test_df = create_dataframes(images_set)
    
    test_gen = create_image_generators(test_df)

    # Load model
    model_path = askopenfilename(
        initialdir=MODELS_DIRECTORY,
        title="Select a .keras model",
        filetypes=[("Keras Model Files", "*.keras")]
    )
    if model_path:
        print("File selected: " + model_path)
    else:
        print("No file selected.")
        sys.exit()

    model = load_model(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_og = os.path.basename(model_path).split('.')[0]

    model_directory = os.path.join(MODELS_DIRECTORY, model_og)
    test_directory = os.path.join(model_directory, "test", DATE)
    os.makedirs(test_directory, exist_ok=True) #models/model_name/date/test/date/

    log_path = os.path.join(test_directory, "testing_log.txt")
    with open(log_path, "w") as log_file:
        sys.stdout = Tee(log_file)
    
        print("\nLabels Distribution:")
        print(test_df["classes"].value_counts())
        
        print("Evaluating model on test data...")
        _, test_acc = model.evaluate(test_gen, verbose=1)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        pred = model.predict(test_gen)
        pred = np.argmax(pred, axis=1) #pick class with highest  probability

        labels = (test_gen.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        pred2 = [labels[k] for k in pred]

        y_test = test_df.classes # set y_test to the expected output
        print(classification_report(y_test, pred2))
        print("Accuracy of the Model:","{:.1f}%".format(accuracy_score(y_test, pred2)*100))

        confusion(test_gen, y_test, pred2, model_name, test_directory)

    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()