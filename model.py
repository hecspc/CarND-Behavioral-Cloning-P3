import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import keras

BATCH_SIZE = 128
EPOCHS=150
# We use the data provided by Udacity
DATA_PATH = './data/provided'
images_path = DATA_PATH + '/IMG/'
steering_correction = 0.25
samples = []
rows_dropped = 0
with open(DATA_PATH + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip the header
    for row in reader:
        samples.append(row)


from sklearn.model_selection import train_test_split
# from the samples we use 20% for validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def loadImage(image_name, doFlip):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR) # Return BGR
    if doFlip:
        image = cv2.flip(image, 1)
    # Convert to YUV
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image

def add_random_shadow(image):
    # image is in YUV
    top_y = 320*np.random.uniform()
    top_x = 0
    bottom_x = 160
    bottom_y = 320*np.random.uniform()
    shadow_mask = np.zeros(image.shape[:2])
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bottom_y-top_y) -(bottom_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image[:,:,0][cond1] = image[:,:,0][cond1]*random_bright
        else:
            image[:,:,0][cond0] = image[:,:,0][cond0]*random_bright

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
    height_shift = np.random.randint(-10,10)
    angle += shift*0.004
    angle = np.clip(angle, -1,1)
    h,w,ch = image.shape
    pts1 = np.float32([[20,0],[w-20,0],[20,h/2],[w-20,h/2]])
    pts2 = np.float32([[20+shift,0],[w-20+shift,0],[20+shift,h/2+height_shift],[w-20+shift,h/2+height_shift]])
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
from keras.utils import plot_model

train_generator = generator(train_samples, batch_size=BATCH_SIZE, is_training=True)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, is_training=False)

model = Sequential()
model.add(Cropping2D(cropping=((55,25), (20, 20)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5-1.0))
model.add(Convolution2D(24, (5, 5), strides=(2,2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation="relu"))
model.add(Convolution2D(48, (5, 5), strides=(2,2), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="tanh", use_bias=False))

print(model.summary())
plot_model(model, to_file='model.png')

# Callback for Tensorboard visualization
tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./log', histogram_freq=1, write_graph=True, write_images=True,
    embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None)
# Saving the model when the validation loss get lower
checkpointCallback = keras.callbacks.ModelCheckpoint(
    'model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='auto', period=1)
# Reduce the learning rate if validation loss is stuck
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,
    patience=3, min_lr=0.0001)
# Stop if validation loss does not reduce
earlyStopCallBack = keras.callbacks.EarlyStopping(monitor='val_loss',
    min_delta=0.0005, patience=5, verbose=0, mode='auto')

model.compile(optimizer=Adam(lr=0.001), loss='mse')

# history_object = model.fit_generator(train_generator, steps_per_epoch =
#     len(train_samples) * 7 / BATCH_SIZE, validation_data =
#     validation_generator,
#     validation_steps = len(validation_samples) / BATCH_SIZE,
#     epochs=EPOCHS, verbose=1,
#     callbacks=[tbCallBack, checkpointCallback, reduce_lr, earlyStopCallBack])

# model.save('model.h5')
#

def plot_random_image():
    batch_sample = samples[np.random.randint(len(samples))]
    img_center = loadImage(images_path + batch_sample[0].split('/')[-1], False)
    img_left = loadImage(images_path + batch_sample[1].split('/')[-1], False)
    img_right = loadImage(images_path + batch_sample[2].split('/')[-1], False)
    img_center_flipped = loadImage(images_path + batch_sample[0].split('/')[-1], True)
    img_left_flipped = loadImage(images_path + batch_sample[1].split('/')[-1], True)
    img_right_flipped = loadImage(images_path + batch_sample[2].split('/')[-1], True)
    img_shift, steering_shift = randomModification(img_center, 0)
    fig, ax = plt.subplots(nrows=4, ncols=1)
    ax.axis("off")
    ax.plot(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    ax.plot(cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB))
    ax.plot(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    ax.plot(cv2.cvtColor(img_left_flipped, cv2.COLOR_BGR2RGB))
    ax.plot(cv2.cvtColor(img_center_flipped, cv2.COLOR_BGR2RGB))
    ax.plot(cv2.cvtColor(img_right_flipped, cv2.COLOR_BGR2RGB))
    ax.plot(cv2.cvtColor(img_shift, cv2.COLOR_BGR2RGB))
    fig.savefig('images/traning_set.png')
    plt.show()
plot_random_image()
exit()
