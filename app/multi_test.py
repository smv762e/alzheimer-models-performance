# type: ignore
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Import configurations
from config import IMG_SHAPE, CLASS_NAMES

def multi_test_func(single_image, mod):
    final_msg = "⚠️ An error occurred during prediction."

    # Load the model
    try:
        model = load_model(mod)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit()

    # Preprocess the image as done during training
    img = single_image.resize((IMG_SHAPE[0], IMG_SHAPE[1]))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if applied during training

    # Predict
    preds = model.predict(img_array)[0]
    pred_index = np.argmax(preds)
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(preds[pred_index])

    final_msg = (
        f"✅ Prediction completed.\n"
        f"🧠 Predicted Class: {pred_class}\n"
        f"📈 Confidence: {confidence * 100:.2f}%\n"
        f"📊 All Class Probabilities:\n" +
        "\n".join([f"- {label}: {float(prob) * 100:.2f}%" for label, prob in zip(CLASS_NAMES, preds)])
    )
    
    return final_msg