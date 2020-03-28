import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
np.random.seed(0)
import pickle
import pandas as pd
import random
import cv2
import requests
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

#converts a colored image into a grey image
def grayscale(img):
    newImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return newImg

#HISTOGRAM EQUALIZATION
#It helps extract features on the image but adding contrasts
def equalize(img):
    newImg = cv2.equalizeHist(img)
    return newImg

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def leNet_model():
    model = Sequential()
    #Conv2D parameters: 30 filters, 5x5 filter size, shape on input image
    #this results in an output of 780 parameters after applying filters
    #results also into an image of shape 28x28x60
    model.add(Conv2D(60, (5,5), input_shape=(32, 32, 1), activation='relu'))
    #results into images of shape 24x24x60
    model.add(Conv2D(60, (5,5), activation='relu'))
    #pooling layers
    #by applying a filter of 2x2, scales down the image to 12x12x60
    model.add(MaxPooling2D(pool_size=(2,2)))

    #results into an image shape of 10x10x30
    model.add(Conv2D(30, (3,3), activation='relu'))
    #results into images of shape 8x8x30
    model.add(Conv2D(30, (3,3), activation='relu'))
    #results in a image shape of 4x4x30
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    #6x6x15 = 540 nodes
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    #drops half of the inpt nodes at each update
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics =['accuracy'])
    return model

#opens a file in binary mode
with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('german-traffic-signs/valid.p', 'rb') as f:
    valid_data = pickle.load(f)
with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

#creates arrays with images of traffic signs loaded from the files
X_train, y_train = train_data['features'], train_data['labels']
X_valid, y_valid = valid_data['features'], valid_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(X_valid.shape[0] == y_valid.shape[0]), "The number of images is not equal to the number of labels."
assert(X_valid.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."

data = pd.read_csv('german-traffic-signs/signnames.csv')

#a map function. for each element in x_train, it will apply the preprocessing function.
# the result of this is all X_train images are now grey and it applied constrast, also normalized
X_train = np.array(list(map(preprocessing, X_train)))
X_valid = np.array(list(map(preprocessing, X_valid)))
X_test = np.array(list(map(preprocessing, X_test)))

#parameters: number of images, height, weight, depth
X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_valid = X_valid.reshape(4410, 32, 32, 1)

#fit ImageDataGenerator

#creates a specification to generate new images accordingly to the parameters below
datagen = ImageDataGenerator(width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.2,
                               shear_range=0.1,
                               rotation_range=10)

#targets the images we want to modify
datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20)
#it actually generates 20 new images based on the datagen definition
X_batch, y_batch = next(batches)

# creates the one hot encoding values for the labels, 43 is the number of classes that are necessary to classify
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_valid = to_categorical(y_valid, 43)

model = leNet_model()
print(model.summary())

#model.fit(X_train, y_train, epochs = 10, validation_data=(X_valid, y_valid), batch_size = 400, verbose = 1, shuffle = 1)

model.fit_generator(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch=2000, epochs=10, validation_data=(X_valid, y_valid), shuffle=1)

score = model.evaluate(X_test, y_test, verbose = 0)

url = 'https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)

img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
img = img.reshape(1, 32, 32, 1)

print("predicted sign: " + str(model.predict_classes(img)))
