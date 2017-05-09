import csv
import cv2
import numpy as np

car_images = []
steering_angles = []

images_path = './data/IMG/'
steering_correction = 0.2

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip the header
    for row in reader:
        img_center = cv2.imread(images_path + row[0].split('/')[-1])
        img_left = cv2.imread(images_path + row[1].split('/')[-1])
        img_right = cv2.imread(images_path + row[2].split('/')[-1])

        car_images.extend([img_center, img_left, img_right])

        steering_center = float(row[3])
        steering_left = steering_center + steering_correction
        steering_right = steering_center - steering_correction
        steering_angles.extend([steering_center, steering_left, steering_right])


X_train = np.array(car_images)
y_train = np.array(steering_angles)

print(X_train.shape)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')

exit()
