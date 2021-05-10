# Part 4 - Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image


def predict_img(img,model):
    test_image = image.load_img(img, color_mode="grayscale", target_size=(28, 28))
    test_image = image.img_to_array(test_image)
    test_image = test_image.reshape(1, 28, 28, 1)
    test_image = test_image.astype('float32')
    test_image = test_image / 255
    result = model.predict(test_image)
    class_prediction = np.argmax(result)

    # Map apparel category with the numerical class
    if class_prediction == 0:
        product = "T-shirt/top"
    elif class_prediction == 1:
        product = "Trouser"
    elif class_prediction == 2:
        product = "Pullover"
    elif class_prediction == 3:
        product = "Dress"
    elif class_prediction == 4:
        product = "Coat"
    elif class_prediction == 5:
        product = "Sandal"
    elif class_prediction == 6:
        product = "Shirt"
    elif class_prediction == 7:
        product = "Sneaker"
    elif class_prediction == 8:
        product = "Bag"
    else:
        product = "Ankle boot"

    return product
