import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import cv2
import imutils

# check if the library is loaded
print("OpenCV version: {}".format(cv2.__version__))
print("Keras version: {}".format(keras.__version__))
print("Numpy version: {}".format(np.__version__))
print("Imutils version: {}".format(imutils.__version__))
print("Tensorflow version: {}".format(tf.__version__))