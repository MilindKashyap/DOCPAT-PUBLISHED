# c.py
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from pathlib import Path

def predict_image(file_path):
    try:
        # Load the pre-trained model (portable path for Linux/Windows)
        base_dir = Path(__file__).resolve().parent
        model_path = str(base_dir / 'model' / 'CNN_model.h5')
        print(f"Loading model from: {model_path}")
        loaded_model = load_model(model_path)  # Load the model
        print("Model loaded successfully!")

        # Preprocess the image
        img = load_img(file_path, target_size=(224, 224))  # Resize to target size
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values

        # Debugging: Check the shape and content of the image array
        print(f"Image shape after preprocessing: {img_array.shape}")

        # Predict the class
        predictions = loaded_model.predict(img_array)
        print(f"Raw predictions: {predictions}")

        # Get the predicted class
        predicted_class = np.argmax(predictions, axis=1)[0]
        print(f"Predicted class: {predicted_class}")

        return predicted_class  # Return only the predicted class
    except Exception as e:
        print(f"Error processing image or predicting: {e}")
        return None
