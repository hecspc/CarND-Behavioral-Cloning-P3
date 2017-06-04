import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import keras

BATCH_SIZE = 128
EPOCHS=150
DATA_PATH = './data/provided'
images_path = DATA_PATH + '/IMG/'
steering_correction = 0.25
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

def loadImage(image_name, doFlip):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR) # Return BGR
    if doFlip:
        image = cv2.flip(image, 1)
    # Convert to YUV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image

def randomModification(image, angle):
    # Image is in YUV
    if np.random.randint(5) == 0:
        return image, angle
    # random blur
    image = cv2.GaussianBlur(image, (np.random.randint(2)*2+1,np.random.randint(2)*2+1), 0)
    # random brightness
    image[0,:,:] = np.clip(image[0,:,:] + np.random.randint(-30,30), 0, 255)
    # random shadow
    image = add_random_shadow(image)
    # add warp
    assert image.shape == (160, 320, 3)
    shift = np.random.randint(-20,20)
    shift2 = shift #np.random.randint(-20,20)
    height_shift = np.random.randint(-10,10)
    # shift is how much to move the bottom row left and right
    # shift2 is how much to move the middle left and right
    # so if shift=shift2, then we are just translating the image
    angle += (shift+shift2)*0.002
    angle = np.clip(angle, -1,1)
    h,w,ch = image.shape
    pts1 = np.float32([[20,0],[w-20,0],[20,h/2],[w-20,h/2]])
    pts2 = np.float32([[20+shift,0],[w-20+shift,0],[20+shift2,h/2+height_shift],[w-20+shift2,h/2+height_shift]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return image, angle  
    
def generator(samples, batch_size=BATCH_SIZE, is_training=False):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                img_center = loadImage(images_path + batch_sample[0].split('/')[-1], False)
                img_left = loadImage(images_path + batch_sample[1].split('/')[-1], False)
                img_right = loadImage(images_path + batch_sample[2].split('/')[-1], False)
                img_center_flipped = loadImage(images_path + batch_sample[0].split('/')[-1], True)
                img_left_flipped = loadImage(images_path + batch_sample[1].split('/')[-1], True)
                img_right_flipped = loadImage(images_path + batch_sample[2].split('/')[-1], True)
                
                steering_center = float(batch_sample[3])
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction
                
                car_images.extend([img_center, img_left, img_right, img_center_flipped, img_left_flipped, img_right_flipped])
                steering_angles.extend([steering_center, steering_left, steering_right, -steering_center, -steering_left, -steering_right])
                if is_training:
                    img_shift, steering_shift = randomModification(img_center, steering_center)
                    car_images.extend([img_center, img_left, img_right, img_center_flipped, img_left_flipped, img_right_flipped, img_shift])
                    steering_angles.extend([steering_center, steering_left, steering_right, -steering_center, -steering_left, -steering_right, steering_shift])
                else :
                    car_images.extend([img_center])
                    steering_angles.extend([steering_center])
                
            X_train = np.array(car_images, dtype=np.float64)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, SpatialDropout2D, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
# from keras.utils import plot_model

train_generator = generator(train_samples, batch_size=BATCH_SIZE, is_training=True)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, is_training=False)


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
model.add(Lambda(lambda x: x/127.5-1.0))
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

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./log', histogram_freq=1, write_graph=True, write_images=True,
    embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None)
checkpointCallback = keras.callbacks.ModelCheckpoint(
    'model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='auto', period=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,
    patience=3, min_lr=0.0001)
earlyStopCallBack = keras.callbacks.EarlyStopping(monitor='val_loss', 
    min_delta=0.0005, patience=5, verbose=0, mode='auto')
    
model.compile(optimizer=Adam(lr=0.001), loss='mse')

from keras.models import load_model
# model = load_model('model.h5')
history_object = model.fit_generator(train_generator, steps_per_epoch =
    len(train_samples) * 7 / BATCH_SIZE, validation_data =
    validation_generator,
    validation_steps = len(validation_samples) / BATCH_SIZE,
    epochs=EPOCHS, verbose=1,
    callbacks=[tbCallBack, checkpointCallback, reduce_lr, earlyStopCallBack])

model.save('model.h5')


plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
