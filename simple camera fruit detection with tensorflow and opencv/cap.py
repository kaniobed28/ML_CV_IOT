import cv2 as cv
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from fruit_predict import predict_image




def vidCap(cameranum):
    cam = cv.VideoCapture(cameranum)
    while True:
        ret,frame = cam.read()
        cv.imshow("window",frame)

        if cv.waitKey(1) == ord('q'):
            break
        elif cv.waitKey(1) == ord('s'):
            name = f" {time.time()}screenshot.png"
            cv.imwrite(name,frame)
            loaded_model = tf.keras.models.load_model('my_model.h5')

            # Specify the path to the image you want to predict
            image_path_to_predict = name

            # Set the custom image size
            custom_image_size = (240, 240,3)  # Replace with the size you used during training

            # Use the function to make a prediction
            predicted_class = predict_image(loaded_model, image_path_to_predict, custom_image_size)
            print("Predicted class:", predicted_class)
    cam.release()
    cv.destroyAllWindows()
    return name

vidCap(0)

