import pandas as pd
import matplotlib.pyplot as plt
import os

# Dictionary of model names and their corresponding training history paths
csv_paths = {
    "Inception": "models/Inception/2025-05-04_11-31/training_history.csv",
    "ResNet50": "models/ResNet50/2025-05-04_11-47/training_history.csv",
    "ResNet50V2": "models/ResNet50V2/2025-05-04_12-02/training_history.csv",
    "ResNet101": "models/ResNet101/2025-05-04_12-17/training_history.csv",
    "ResNet101V2": "models/ResNet101V2/2025-05-04_12-40/training_history.csv",
    "ResNet152": "models/ResNet152/2025-05-04_13-05/training_history.csv",
    "ResNet152V2": "models/ResNet152V2/2025-05-04_13-34/training_history.csv",
    "VGG16": "models/VGG16/2025-05-08_16-18/training_history.csv",
    "VGG19": "models/VGG19/2025-05-04_14-41/training_history.csv",
    "Xception": "models/Xception/2025-05-04_15-12/training_history.csv"
}

# Output file for the final figure
output_file = "app\helper\models_training_metrics.png"


# Line styles to differentiate models
line_styles = ['-', '--', '-.', ':'] * 3  # Up to 12 styles

# Metrics to plot
metrics = ["loss", "accuracy", "val_loss", "val_accuracy"]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# Plot each metric
for i, metric in enumerate(metrics):
    ax = axes[i]
    for j, (model_name, csv_path) in enumerate(csv_paths.items()):
        df = pd.read_csv(csv_path)
        if metric in df.columns:
            ax.plot(df[metric], line_styles[j % len(line_styles)], label=model_name)

    ax.set_title(f"{metric.replace('_', ' ').capitalize()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace('_', ' ').capitalize())
    ax.grid(True)
    ax.legend(fontsize='small')

plt.suptitle("Training Metrics Evolution by Model", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_file)
plt.close()
