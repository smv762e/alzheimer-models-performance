# type: ignore
import os
import sys
from datetime import datetime
import pandas as pd
import gradio as gr

# Import custom modules
from src.data_utils import create_dataframes, create_image_generators
from src.log_utils import Tee
from src.model_utils import select_model_by_name, build_model, create_callbacks, plot_training_history

# Import configurations
from config import MODELS_DIRECTORY

def train_val_func(train_set, val_set, mod, num_epochs):
    final_msg = "⚠️ An error occurred during training."
    history_df = None
    image_path = None

    try:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Read Train and Val .csv -> Dataframe
        train_df = pd.read_csv(train_set)
        val_df = pd.read_csv(val_set)

        if train_df.empty or val_df.empty:
            raise gr.Error("❌ Training or validation set is empty.")

        num_classes = train_df['classes'].nunique()
        if num_classes < 2:
            raise gr.Error("❌ Not enough classes for training.")

        # Create image generators
        train_gen = create_image_generators(train_df)
        print(train_gen.class_indices)
        val_gen = create_image_generators(val_df)

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
            print("\nTraining Set:")
            print(train_df["classes"].value_counts())
            print("\nValidation Set:")
            print(val_df["classes"].value_counts())

            # Save model summary to a separate file
            model.summary(print_fn=lambda x: summary_file.write(x + "\n"))

            # Create callbacks and train the model
            callbacks = create_callbacks(model_name, model_directory)
            print("\n💡 Training in progress...")
            history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, callbacks=callbacks)

            # Save training history
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(os.path.join(model_directory, 'training_history.csv'), index=False)
            history_df['lr'] = history_df['lr'].apply(lambda x: f"{x:.5f}")
            history_df = history_df.round({'loss':5, 'accuracy':5, 'val_loss':5, 'val_accuracy':5})

            # Plot training history
            image_path = plot_training_history(history, model_name, model_directory)

            final_msg = (
                f"✅ Training and validation completed.\n"
                f"📊 Labels Distribution:\n"
                f"Training Set:\n"
                f"{train_df['classes'].value_counts().to_string()}\n"
                f"Validation Set:\n"
                f"{val_df['classes'].value_counts().to_string()}\n"
            )
            
    except gr.Error as ge:
        raise ge  # Errores de Gradio los dejamos pasar
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise gr.Error(f"❌ Unexpected error: {str(e)}")
    finally:
        sys.stdout = sys.__stdout__
        return final_msg, history_df, image_path