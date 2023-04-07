import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import cv2
import imutils

# check if the library is loaded
print("OpenCV version: {}".format(cv2.__version__))
print("Keras version: {}".format(keras.__version__))
print("Numpy version: {}".format(np.__version__))
print("Imutils version: {}".format(imutils.__version__))
print("Tensorflow version: {}".format(tf.__version__))
print("")

# Load the model
print("Loading model...")
model = load_model('handwriting.model')
print("Model loaded")

# Load image
image = cv2.imread("./images/2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

# Find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# keep every coordinate of the contours
chars = []

# loop over the contours
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    # filter out the contours that are too small or too big
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        if tW > tH:
            thresh = imutils.resize(thresh, width=32)
        else :
            thresh = imutils.resize(thresh, height=32)
        
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)
        
        # pad the image and force 32x32 dimensions
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded = cv2.resize(padded, (32, 32))

        # prepare the padded image for classification via our handwriting OCR model
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)
        chars.append((padded, (x, y, w, h)))
      
# loop over the characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")

# classify the characters
preds = model.predict(chars)
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# loop over the predictions and bounding boxes
for (pred, (x, y, w, h)) in zip(preds, boxes):
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    print ("{} - {:.2f}%".format(label, prob * 100))

    # draw the prediction on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

# Show the padded image
cv2.imshow("Image", image)
cv2.waitKey(0)