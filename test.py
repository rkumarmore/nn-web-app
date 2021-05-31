import tensorflow as tf
import numpy as np

image_size = 256
img  = tf.keras.preprocessing.image.load_img(
    'Prediction - Flower.jpg', grayscale=False, color_mode="rgb", interpolation="nearest",
    target_size=(image_size, image_size)
)
input_arr = tf.keras.preprocessing.image.img_to_array(img)
input_arr = np.array([input_arr])
pickled_bot_cnn_model = tf.keras.models.load_model('pickled_bot_cnn_model')
predicted = pickled_bot_cnn_model.predict(input_arr)
classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '2', '3', '4', '5', '6', '7', '8', '9']
classes[np.where(predicted == 1.)[1][0]]
img_class = classes[np.where(predicted == 1.)[1][0]]
print(img_class)