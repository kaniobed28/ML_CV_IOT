import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model, img_path, custom_image_size):
    """
    Predicts the class label for an input image using a trained model.

    Parameters:
    - model: The trained TensorFlow model.
    - img_path: The path to the input image file.
    - custom_image_size: Tuple specifying the target size for the input image.

    Returns:
    - predicted_class: The predicted class label for the input image.
    """
    # Load and preprocess the image for prediction
    img = image.load_img(img_path, target_size=(custom_image_size[0], custom_image_size[1]))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the image data

    # Make a prediction
    predictions = model.predict(img_array)

    # Decode the prediction
    predicted_class = np.argmax(predictions)
    
    return predicted_class

