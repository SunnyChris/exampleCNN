import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import time
from datetime import datetime

log_dir = os.path.join(
    "logs",
    "fit",
    datetime.now().strftime("%Y%m%d-%H%M%S"),
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

##DEFINE DATA
DATADIR = "C:/TEST/Images"
CATEGORIES = ["A", "B", "C", "D", "E", "F"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # path to categories dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        y = img_array.shape[0]
        x = img_array.shape[1]  
        new_array = img_array[150:750,150:750]
        plt.imshow(new_array, cmap= 'gray')
        plt.show()
        break
    break


IMG_SIZE = 100

new2_array = cv2.resize(new_array, (IMG_SIZE, IMG_SIZE)) 
plt.imshow(cv2.cvtColor(new2_array, cv2.COLOR_BGR2RGB))
plt.show()

##TRAINING DATA
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                
                y = img_array.shape[0]
                x = img_array.shape[1]
                new_array = img_array[150:750,150:750]
                
                new2_array = cv2.resize(new_array, (IMG_SIZE, IMG_SIZE))
                
                training_data.append([new2_array, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))



import random

random.shuffle(training_data)


for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
y = to_categorical(y)

X = X/255.0

## BUILD MODEL
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())#Conv is 2D and Dense layer needs 1D Data
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", #Different models: binary_crossentropy, sparse_categorical_crossentropy, categorical_crossentropy
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, batch_size=5, epochs=100, validation_split=0.1, callbacks=[tensorboard_callback])
