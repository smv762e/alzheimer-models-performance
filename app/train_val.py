# type: ignore
import os
import sys
from datetime import datetime
import pandas as pd

# Import custom modules
from src.data_utils import create_dataframes, split_data, create_image_generators
from src.log_utils import Tee
from src.model_utils import select_model_by_name, build_model, create_callbacks, plot_training_history

# Import configurations
from config import MODELS_DIRECTORY

def train_val_func(images_set, mod, num_epochs):
    final_msg = "⚠️ An error occurred during training."
    image_path = None

    try:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Create DataFrame with images
        data_df = create_dataframes(images_set)
        num_classes = data_df['classes'].nunique()

        # Split into train, validation, and test
        train_df, val_df, test_df = split_data(data_df)

        # Create image generators
        train_gen = create_image_generators(train_df)
        print(train_gen.class_indices)
        val_gen = create_image_generators(val_df)
        test_gen = create_image_generators(test_df)

        # Build model
        model_fn, model_name = select_model_by_name(mod)
        model = build_model(model_fn, num_classes)

        # Create directory to save the model
        model_directory = os.path.join(MODELS_DIRECTORY, model_name, date_str)
        os.makedirs(model_directory, exist_ok=True)

        # Configure output redirection
        log_path = os.path.join(model_directory, "training_log.txt")
        summary_path = os.path.join(model_directory, "model_summary.txt")

        with open(log_path, "w", encoding="utf-8") as log_file, \
            open(summary_path, "w", encoding="utf-8") as summary_file:

            sys.stdout = Tee(log_file)

            print("\n📊 Labels Distribution:")
            print(data_df["classes"].value_counts())

            # Save model summary to a separate file
            model.summary(print_fn=lambda x: summary_file.write(x + "\n"))

            # Create callbacks and train the model
            callbacks = create_callbacks(model_name, model_directory)
            print("\n💡 Training in progress...")
            history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, callbacks=callbacks)

            # Save training history
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(os.path.join(model_directory, 'training_history.csv'), index=False)

            # Plot training history
            image_path = plot_training_history(history, model_name, model_directory)

            # Final model evaluation
            print("🔍 Evaluating model on test data...")
            _, test_acc = model.evaluate(test_gen, verbose=1)
            print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")

            final_msg = (
                "✅ Training and validation completed.\n"
                "📊 Labels Distribution:\n"
                f"{data_df['classes'].value_counts().to_string()}\n"
                f"🎯 Test Accuracy: {test_acc * 100:.2f}%"
            )
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")

    finally:
        sys.stdout = sys.__stdout__
        return final_msg, image_path