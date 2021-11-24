# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:12:12 2021

@author: pablo
"""

%reset

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Assign which GPU to run on, "0" or "1"
import tensorflow as tf
import scipy.io as sio
from tensorflow import keras
tf.keras.backend.clear_session()
import numpy as np
import scipy.io as sio
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, concatenate,Flatten,Reshape,Cropping2D
from tensorflow.python.keras.models import Model
import matplotlib as mpl

#------------------------------------------------------------------------------------------------------------------------
# To define
#------------------------------------------------------------------------------------------------------------------------
intensity_level = 5
background = 0.01
Directory = 'C:/Users/pablo/Documents/Iron images/'

Training_Dataset_path = os.path.join(Directory, 'data3.mat')
Training_Dataset_path_noise = os.path.join(Directory, 'data_noise3.mat')
checkpoint_path = os.path.join(Directory, 'Checkpoint','Checkpoint_REPRODUCED_model6layers_'+str(intensity_level)+'_'+str(background)+'/cp.ckpt')

#Nepochs = 150
Nepochs =1
#------------------------------------------------------------------------------------------------------------------------
# Load the data - Training (normalised)
#------------------------------------------------------------------------------------------------------------------------
X_training = sio.loadmat(Training_Dataset_path)
X_training = X_training['saving']
X_training_noisy = sio.loadmat(Training_Dataset_path_noise)
X_training_noisy = X_training_noisy['saving_noise']
    
    
xtr = np.zeros((16000, 60, 60, 1))
xtr[:,:,:,0]=X_training[0:16000,:,:]

xtr_noise = np.zeros((16000, 60, 60, 1))
xtr_noise[:,:,:,0]=X_training_noisy[0:16000,:,:]

X_validation_noisy= np.zeros((200, 60, 60, 1))
X_validation_noisy[:,:,:,0]=xtr_noise[0:200,:,:,0]

X_validation= np.zeros((200, 60, 60, 1))
X_validation[:,:,:,0]=xtr[0:200,:,:,0]

mpl.pyplot.imshow(xtr[3999,:,:,0])

#X = np.swapaxes(X, 0, -1)
#X = np.swapaxes(X, 1, 2)
#X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
#X_noisy = np.zeros((X.shape[0], X.shape[1], X.shape[2], 1))

#for index in range(X.shape[0]):
    #image = np.reshape(X[index,:,:,:], (X.shape[1], X.shape[2]))
    #max_image, min_image = np.amax(image), np.amin(image)
    #image = (image - min_image)/(max_image - min_image)
    #I_Poisson = intensity_level*image + background
    #I_Poisson = np.array(I_Poisson)
    #I_Poisson = I_Poisson.astype(float)
    #I_Poisson_noisy = np.random.poisson(I_Poisson)

    #max_noisy, min_noisy = np.amax(I_Poisson_noisy), np.amin(I_Poisson_noisy)
    #I_Poisson_noisy = (I_Poisson_noisy - min_noisy) / (max_noisy - min_noisy)
    #X_noisy[index, :, :, :] = np.reshape(I_Poisson_noisy, (I_Poisson_noisy.shape[0], I_Poisson_noisy.shape[1], 1))
    #X[index, :, :, :] = np.reshape(image, (image.shape[0], image.shape[1], 1))

#X = X[:201]
#X_noisy = X_noisy[:10000]
#X_training = X[1000:]
#X_training_noisy = X_noisy[1000:]
#X_validation = X[:1000]
#X_validation_noisy = X_noisy[:1000]

#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
def create_model():
    UB = True
    w = 5
    inputs = Input((60,60,1))
    #inputs = Input((None,None,1))
    conv1 = Conv2D(64, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(inputs)
    pool1 = MaxPool2D((2,2), padding='same')(conv1)

    conv2 = Conv2D(128, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool1)
    pool2 = MaxPool2D((2,2), padding='same')(conv2)

    conv3 = Conv2D(256, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool2)
    #pool3 = MaxPool2D((2,2), padding='same')(conv3)

    #conv4 = Conv2D(512, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool3)
    # = MaxPool2D((2,2), padding='same')(conv4)

    #conv5 = Conv2D(1024, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool4)
   # pool5 = MaxPool2D((2,2), padding='same')(conv5)
    
    #conv6 = Conv2D(2048, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool5)
    #pool6 = MaxPool2D((2,2), padding='same')(conv6)

    #conv7 = Conv2D(4096, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool6)


    #deconv6 = Conv2DTranspose(2048, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(conv7)
    #cropped_decon6 = Cropping2D(cropping=((0, 0), (1, 0)))(deconv6)
   # merge6 = concatenate([conv6,cropped_decon6], axis = 3)

    #deconv5 =  Conv2DTranspose(1024, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge6)
    #cropped_decon5 = Cropping2D(cropping=((1, 0), (1, 0)))(deconv5)
    #merge5 = concatenate([conv5,cropped_decon5], axis = 3)
    #deconv4 = Conv2DTranspose(512, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge5)
    #cropped_decon4 = Cropping2D(cropping=((1, 0), (1, 0)))(deconv4)
    #merge4 = concatenate([conv4,cropped_decon4], axis = 3)

    #deconv3 = Conv2DTranspose(256, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge4)
    #cropped_deconv3 = Cropping2D(cropping=((1, 0), (0, 0)))(deconv3)
    #merge3 = concatenate([conv3,cropped_deconv3], axis = 3)

    deconv2 = Conv2DTranspose(128, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(conv3)
    merge2 = concatenate([conv2,deconv2], axis = 3)
    deconv1 = Conv2DTranspose(64, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge2)
    merge1 = concatenate([conv1,deconv1], axis = 3)
    conv_last = Conv2D(1, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(merge1)

    model = Model(inputs,conv_last)

    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00010), loss = "mean_squared_error", metrics = ["accuracy"])
    
    return model

model = create_model()

model.summary()



checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

start_time = time.time()
history = model.fit(xtr_noise, xtr, validation_data = (X_validation_noisy, X_validation), epochs = Nepochs, batch_size = 32, callbacks=[cp_callback])
training_time = time.time()-start_time

#------------------------------------------------------------------------------------------------------------------------
# Test & Save on validation data
#------------------------------------------------------------------------------------------------------------------------
X_val_noisy = X_validation_noisy[0:5]
X_val = X_validation[0:5]
model.load_weights(checkpoint_path)

start_time = time.time()
predict = model.predict(X_val_noisy)

test2=xtr_noise[200,:,:,0]
predict2= model.predict(xtr_noise[100:101,:,:,:])



aa=predict2[0,:,:,0]

mpl.pyplot.imshow(predict2[0,:,:,0])
mpl.pyplot.imshow(xtr_noise[200,:,:,:])
mpl.pyplot.imshow(xtr[200,:,:,:])

testing_time = time.time()-start_time
dictionary = {}
dictionary['simulated'] = {'input':X_val_noisy, 'predictions':predict, 'GT':X_val}
dictionary['time'] = {'training_time': training_time, 'testing_time': testing_time}

sio.savemat(os.path.join(Directory, 'Simulated_Data/res_simulated_6layers_'+str(intensity_level)+'_'+str(background)+'.mat'), dictionary)
sio.savemat(os.path.join(Directory, 'Simulated_Data/history_'+str(intensity_level)+'_'+str(background)+'.mat'), history.history)
#© 2021 GitHub, Inc.
#Terms
#Privacy
#Security
#Status
#Docs
#Contact GitHub
#Pricing
#API
#Training
#Blog
#About
