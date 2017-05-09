import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

images_path = './data/IMG/'
steering_correction = 0.2
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip the header
    for row in reader:
        samples.append(row)



from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                img_center = cv2.imread(images_path + batch_sample[0].split('/')[-1])
                img_left = cv2.imread(images_path + batch_sample[1].split('/')[-1])
                img_right = cv2.imread(images_path + batch_sample[2].split('/')[-1])

                car_images.extend([img_center, img_left, img_right])

                steering_center = float(batch_sample[3])
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction
                steering_angles.extend([steering_center, steering_left, steering_right])
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0, 0))))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
#
history_object = model.fit_generator(train_generator, steps_per_epoch =
    len(train_samples), validation_data =
    validation_generator,
    validation_steps = len(validation_samples),
    epochs=5, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
