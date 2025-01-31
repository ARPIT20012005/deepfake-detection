import tensorflow as tf
import cv2
import numpy as np


def load_model():
    model_path = "deepfake_detection_xception_180k_14epochs.h5"  
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256)) 
    image = image.astype('float32') / 255.0  
    image = np.reshape(image, (1, 256, 256, 3))  
    return image


def predict_image(image_path, model):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction
