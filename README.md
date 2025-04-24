# 🧠 Alzheimer Prediction App

This Gradio-based application provides a full pipeline for image-based Alzheimer’s prediction. It offers tools to create image datasets, train deep learning models, evaluate performance, and make individual predictions.

---

## 🗂️ Tabs Overview

### 📌 1. Introduction

Provides a general welcome message and overview of the application’s purpose.

> **Welcome message:**  
> _"🧠 Welcome to the Alzheimer prediction app.  
> Use the tabs to create datasets, train models, and make predictions."_

---

### 🏗️ 2. Batch Creator

Creates a new dataset from a folder of Alzheimer-classified images.

**Inputs:**
- 📁 Select image folder (via `FileExplorer`)
- 🏷️ Name of the new dataset
- 🔢 Number of images per class

**Outputs:**
- Text summary of dataset creation

---

### 🧪 3. Training and Validation

Trains a deep learning model using selected images and tracks performance.

**Inputs:**
- 📁 Select training image folder
- 🧠 Choose a model (Inception, ResNet, VGG, Xception)
- 🔁 Number of training epochs

**Outputs:**
- 📄 Training summary
- 📊 Training history as a table (loss, accuracy, etc.)
- 🖼️ Accuracy/Loss curve as an image

---

### 🧾 4. Model Evaluation

Evaluates the trained model using a new dataset.

**Inputs:**
- 📁 Select test image folder
- 📦 Select `.keras` trained model file

**Outputs:**
- 📄 Evaluation summary
- 🖼️ Confusion matrix image

---

### 🧍 5. Single Predictions

Predicts the class of a single image using a trained model.

**Inputs:**
- 🖼️ Upload a single image
- 📦 Select trained model

**Outputs:**
- 📄 Predicted class and confidence level

---

## ▶️ How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python alzheimer_app.py
```

The interface will open automatically in your browser.