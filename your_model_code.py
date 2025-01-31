import tensorflow as tf
import cv2
import numpy as np

# Step 1: Load the model
def load_model():
    model_path = "deepfake_detection_xception_180k_14epochs.h5"  # Replace with your model file path
    model = tf.keras.models.load_model(model_path)
    return model

# Step 2: Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Resize image to 256x256 as required by the model
    image = image.astype('float32') / 255.0  # Normalize image
    image = np.reshape(image, (1, 256, 256, 3))  # Reshape to match model input shape
    return image

# Step 3: Predict
def predict_image(image_path, model):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction
