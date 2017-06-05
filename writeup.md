#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run.mp4 video of the car model.h5 on the first track

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run_test
```

####3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 125-141) 

The model includes RELU layers to introduce nonlinearity  and the data is normalized in the model using a Keras lambda layer (code line 127). I also cropped the image using the `Cropping2D` layer to remove the sky and the car from the screenshots (code line 126)


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 136, 138, 140). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py code line 164). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning
The model used an adam optimizer, I also used the the ReduceLROnPlateau callback on Keras to reduce the learning rate when there's no improvement in the validation loss for 3 epochs. (model.py code line 155).


####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolutional network followed by fully connected layers.

My first step was to use a convolution neural network model similar to the Nvidia network model as stated on the paper [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The ratio was to use 20% of my samples data as validation set (model.py line 25). The first run with a model similar to the one on the paper result with a low training loss but a much higher validation loss implying the model was overfitting.

To combat the overfitting, I modified the model so that I add three small dropout layers between the fully connected layers.

To increase the number of samples, I also used the images from the right and left sides of the car applying a constant correction factor to the steering value (model.py lines 90-110). For every image (left, right and center) I also used the flipped images increasing the number of test by a factor of 6. To improve the testing I also added a random modification to the training images to make it more difficult to train.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like the bridge or the turns with no curbs. To improve the driving behavior in these cases, I created more data and I increased the samples for these areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I made a couple of modifications on the `drive.py` script to modify the speed according to desired speed and the angle correction. (drive.py lines 67-78)

####2. Final Model Architecture

The final model architecture (model.py lines 125-141) consisted of a convolution neural network with the following layers and layer sizes ...

```
| Layer                 | Details                              |
|--------------------------------------------------------------|
| Input                 | shape=(160, 320, 3)                  |
| Cropping2D            | cropping=((55, 25), (20, 20))        |
| Lambda                | Normalize between -1 and 1           |
| Conv2D                | filters=24, kernel_size=5, strides=2 |
| Activation relu       |                                      |
| Conv2D                | filters=36, kernel_size=5, strides=2 |
| Activation relu       |                                      |
| Conv2D                | filters=48, kernel_size=5, strides=2 |
| Activation relu       |                                      |
| Conv2D                | filters=64, kernel_size=3            |
| Activation relu       |                                      |
| Conv2D                | filters=64, kernel_size=3            |
| Activation relu       |                                      |
| Conv2D                | filters=64, kernel_size=3            |
| Activation relu       |                                      |
| Flatten               |   				                       |
| Dense                 | 100 units                            |
| Activation relu       |                                      |
| Dropout               | Probability 0.5                      |
| Dense                 | 50 units                             |
| Activation relu       |                                      |
| Dropout               | Probability 0.5                      |
| Dense                 | 20 units                             |
| Activation relu       |                                      |
| Dropout               | Probability 0.2                      |
| Dense                 | 1 unit                               |
| Activation tanh       | No bias                              |
```


![model][model]

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 80, 280, 3)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 80, 280, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 138, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 67, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 32, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 30, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 28, 64)         36928
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, 26, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 1664)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               166500
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 20)                1020
_________________________________________________________________
dropout_3 (Dropout)          (None, 20)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 20
=================================================================
Total params: 340,866
Trainable params: 340,866
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
