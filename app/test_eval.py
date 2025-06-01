# type: ignore
import os
import sys
import numpy as np
from datetime import datetime
import pandas as pd
import gradio as gr
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

# Import custom modules
from src.data_utils import create_image_generators
from src.log_utils import Tee
from src.model_utils import confusion

# Import configurations
from config import MODELS_DIRECTORY

def test_eval_func(test_set, mod):
    final_msg = "⚠️ An error occurred during testing."
    image_path = None

    try:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        # Read Test .csv -> Dataframe
        test_df = pd.read_csv(test_set)

        if test_df.empty:
            raise gr.Error("❌ Test set is empty.")

        test_gen = create_image_generators(test_df)

        # Load model
        try:
            model = load_model(mod)
        except Exception as e:
            raise gr.Error(f"❌ Error loading model: {e}")

        # Create test directory
        model_name = os.path.splitext(os.path.basename(mod))[0]
        model_og = os.path.basename(mod).split('.')[0]
        model_directory = os.path.join(MODELS_DIRECTORY, model_og)
        test_directory = os.path.join(model_directory, "test", date_str)
        os.makedirs(test_directory, exist_ok=True)

        log_path = os.path.join(test_directory, "testing_log.txt")
        with open(log_path, "w", encoding="utf-8") as log_file:

            sys.stdout = Tee(log_file)

            print("\n📊 Labels Distribution:")
            print(test_df["classes"].value_counts())

            print("\n🔍 Evaluating model on test data...")
            _, test_acc = model.evaluate(test_gen, verbose=1)
            print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

            # Get predictions
            pred = model.predict(test_gen)
            if pred.ndim > 1:  # If it's a probability matrix
                pred = np.argmax(pred, axis=1)

            # Convert indices to labels
            labels = {v: k for k, v in test_gen.class_indices.items()}
            pred_labels = [labels[idx] for idx in pred]

            # Convert y_test to list
            y_test = test_df["classes"].tolist()

            # Classification report
            print("\n📜 Classification Report:")
            print(classification_report(y_test, pred_labels))

            print("\n🎯 Accuracy of the Model:", "{:.2f}%".format(accuracy_score(y_test, pred_labels) * 100))

            final_msg = (
                f"✅ Testing evaluation completed.\n"
                f"📊 Labels Distribution:\n"
                f"{test_df['classes'].value_counts().to_string()}\n"
                f"📜 Classification Report:\n"
                f"{classification_report(y_test, pred_labels)}\n"
                f"🎯 Test Accuracy: {test_acc * 100:.2f}%"
            )

            # Generate confusion matrix
            image_path = confusion(test_gen, y_test, pred_labels, model_name, test_directory)

    except gr.Error as ge:
        raise ge
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise gr.Error(f"❌ Unexpected error: {str(e)}")
    finally:
        sys.stdout = sys.__stdout__
        return final_msg, image_path