'''
## Data path acquistion
'''
import csv
import os
import cv2
import numpy as np
from random import shuffle
import imageio
import sklearn
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Cropping2D, Dense, GlobalAveragePooling2D, Flatten, Activation, \
Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization
from keras.models import Model,Sequential
from keras import optimizers
import tensorflow as tf
from math import ceil, floor
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# Data path of track 1
samples = []
with open('./data/Track1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

# Data path of track 1
samples_2 = []        
with open('./data/Track2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample_2 in reader:
        samples_2.append(sample_2)
samples.extend(samples_2)

'''
## Definition of data generator to facility training process
'''

def generator(samples, batch_size=32, multi_cameras = False, filp = False):
    num_samples = len(samples)
    while 1: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steerings = []
            for batch_sample in batch_samples:

                images_temp=[]
                steerings_temp=[]
                filepath_center = batch_sample[0]

                
                image_center = imageio.imread(filepath_center)
                steering_center = float(batch_sample[3])
                images_temp.append(image_center)
                steerings_temp.append(steering_center)
                
                # Use of multiple cameras           
                if multi_cameras is True:
                    filepath_left = batch_sample[1] 
                    filepath_right = batch_sample[2] 
                    
                    # steering angle correction parameters for left and right camera
                    steering_left_cor = 0.1
                    steering_right_cor = 0.105
                    
                    image_left = imageio.imread(filepath_left)
                    image_right = imageio.imread(filepath_right)
                    
                    steering_left = steering_center + steering_left_cor
                    steering_right = steering_center - steering_right_cor
                    
                    images_temp.extend([image_left,image_right])
                    steerings_temp.extend([steering_left,steering_right])
                
                # Image flipping
                if filp is True:
                    # Create filpped frame and opposite sign of steering angle
                    for idx in range(len(images_temp)):
                        image_flipped = np.fliplr(images_temp[idx])
                        steering_flipped = -steerings_temp[idx]
                        images_temp.append(image_flipped)
                        steerings_temp.append(steering_flipped)
                    
                images.extend(images_temp)
                steerings.extend(steerings_temp)
            
            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)

# Separation of data into training and validation samples to feed data generator
train_samples, validation_samples = train_test_split(samples, test_size=0.1)


'''
## Model architecture definition
'''
# Nvidia PilotNet architecture
row,col,ch = 160,320,3
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row,col,ch)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # pixels normalization
model.add(Conv2D(24, (5, 5),strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(36, (5, 5),strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(48, (5, 5),strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam')

'''
# Training
'''
batch_size=64
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size, multi_cameras = True, filp= True)
validation_generator = generator(validation_samples, batch_size=batch_size)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 5)
history = model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=10, verbose=1, callbacks=[es])
'''
Model saving
'''
# model.save('model.h5')