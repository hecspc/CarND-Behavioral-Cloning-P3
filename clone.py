import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import keras

BATCH_SIZE = 64
EPOCHS=5
DATA_PATH = './data/provided'
images_path = DATA_PATH + '/IMG/'
steering_correction = 0.2
samples = []
rows_dropped = 0
with open(DATA_PATH + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip the header
    for row in reader:
        if (True or abs(float(row[3])) >= 0.05 or (abs(float(row[3])) < 0.05 and np.random.randint(10) <= 2)):
            samples.append(row)
        else : 
            rows_dropped += 1
print("Dropped %s rows with low steering"%(rows_dropped))


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=BATCH_SIZE):
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
                img_center_flipped = np.fliplr(img_center)
                img_left_flipped = np.fliplr(img_left)
                img_right_flipped = np.fliplr(img_right)
                
                car_images.extend([img_center, img_left, img_right, img_center_flipped, img_left_flipped, img_right_flipped])

                steering_center = float(batch_sample[3])
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction
                steering_angles.extend([steering_center, steering_left, steering_right, -steering_center, -steering_left, -steering_right])
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, SpatialDropout2D, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
# from keras.utils import plot_model

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


def resize_images(img):
    """Returns resized image
    Cannot be directly used in lambda function
    as tf is not understood by keras
    """
    import tensorflow as tf
    return tf.image.resize_images(img, (66, 200))



model = Sequential()
model.add(Cropping2D(cropping=((55,25), (20, 20)), input_shape=(160,320,3)))
# model.add(Lambda(resize_images))
model.add(Lambda(lambda x: x/255.-0.5))
model.add(Convolution2D(24, (5, 5), strides=(2,2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation="relu"))
model.add(Convolution2D(48, (5, 5), strides=(2,2), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
# model.add(SpatialDropout2D(0.2))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="tanh", use_bias=False))

print(model.summary())
# plot_model(model, to_file='model.png')

# model = make_parallel(model, 2)

model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
#
history_object = model.fit_generator(train_generator, steps_per_epoch =
    len(train_samples) * 6 / BATCH_SIZE, validation_data =
    validation_generator,
    validation_steps = len(validation_samples) * 6 / BATCH_SIZE,
    epochs=EPOCHS, verbose=1)

model.save('model.h5')


plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
