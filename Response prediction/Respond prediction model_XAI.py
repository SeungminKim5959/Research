# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:07:57 2024

@author: NBP Laboratory
"""

import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta
from sklearn import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten,GlobalAveragePooling1D, ZeroPadding1D, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, AveragePooling1D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
import shutil
from tensorflow.compat.v2.keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import lime
import lime.lime_tabular
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, AveragePooling1D, Flatten, Dense, Dropout, Conv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.compat.v2.keras.models import model_from_json
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
plt.interactive(False)
from tensorflow.compat.v2.keras.models import model_from_json
import csv

""" Set Random Seed for Uniform Performance """
tf.random.set_seed(777)
np.random.seed(777)

if 'Dataset_Test' in locals():
    print('Loading Dataset Done')
else: 
        Dataset_Test = np.loadtxt('D:/KSM/TRD/data/240326/Final_Test.csv', dtype=np.float32) 
        Dataset_Train = np.loadtxt('D:/KSM/TRD/data/240326/Second_Train_CV5.csv', dtype=np.float32)
        Dataset_Val = np.loadtxt('D:/KSM/TRD/data/240326/Second_Val_CV5.csv', dtype=np.float32)

     
x_Train = Dataset_Train[:,:800]
y_Train = Dataset_Train[:,800]
x_Val = Dataset_Val[:,:800]
y_Val = Dataset_Val[:,800]
x_Test = Dataset_Test[:,:800]
y_Test = Dataset_Test[:,800]
    
    
shuffle_indices = np.random.permutation(np.arange((len(y_Train))))
x_Train = copy.copy(x_Train[shuffle_indices,:])
y_Train = copy.copy(y_Train[shuffle_indices])

yy_Train = tf.keras.utils.to_categorical(y_Train)
yy_Test = tf.keras.utils.to_categorical(y_Test)
yy_Val = tf.keras.utils.to_categorical(y_Val)

  # Reshape Dataset (For matching with input size)
x_Train = x_Train.reshape((len(x_Train), np.prod(x_Train.shape[1:]),1))
x_Val = x_Val.reshape((len(x_Val), np.prod(x_Val.shape[1:]),1))
x_Test = x_Test.reshape((len(x_Test), np.prod(x_Test.shape[1:]),1))


momentum = 0.9
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

EPOCH = 200

BATCHSIZE = 406
LEARNINGRATE = 0.003
DECAY = 0.4
DO = 0.2
REG = 0.0001
parameters = [BATCHSIZE, LEARNINGRATE, DECAY, DO, REG]

 # Building model
def create_model(input_shape) :
    
    BATCHSIZE = 128
    LEARNINGRATE = 0.35
    DECAY = 0.9
    DO = 0.4
    REG = 0.001
    
    Input = tf.keras.Input(shape=(800,1))
         
    x1 = Conv1D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(REG))(Input)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = AveragePooling1D(pool_size=2, strides=2, padding="same")(x1)
             
    x1 = Conv1D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(REG))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = AveragePooling1D(pool_size=2, strides=2, padding="same")(x1)
         
    x1 = Conv1D(128, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(REG))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = AveragePooling1D(pool_size=2, strides=2, padding="same")(x1)
         
    x1 = Conv1D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(REG))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = AveragePooling1D(pool_size=2, strides=2, padding="same")(x1)
         
    x1 = Conv1D(512, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(REG))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = AveragePooling1D(pool_size=2, strides=2, padding="same")(x1)
         
    x1 = Flatten()(x1)
    x1 = Dense(512, activation='relu', name='fcl1', kernel_regularizer=l2(REG))(x1)
    x1 = Dense(512, name='fcl2', kernel_regularizer=l2(REG))(x1)
    x1 = Activation("relu")(x1)
    x1 = Dropout(DO)(x1)
         
    x1 = Dense(256, activation='relu', name='fcl3', kernel_regularizer=l2(REG))(x1)
    x1 = Dense(256, name='fcl4', kernel_regularizer=l2(REG))(x1)
    x1 = Activation("relu")(x1)
       
    Output = Dense(2, activation='softmax')(x1)
        
    model = tf.keras.Model(Input, Output)
    model.summary()
         
    model.compile(optimizer=RMSprop(lr=LEARNINGRATE, decay=DECAY),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    return model
     
input_shape = x_Test.shape[1:]
#input_shape = tf.keras.Input(shape=(800, 1))
model = create_model(input_shape)

model.load_weights('D:/KSM/TRD/Result/240326/Saved_Model_Weight_2.h5')

from lime import lime_tabular

explainer = lime_tabular.RecurrentTabularExplainer(x_Train,training_labels = yy_Train, feature_names=("SERS"), discretize_continuous=False, feature_selection= 'auto', class_names=['class1', 'class2'])
exp = explainer.explain_instance(np.expand_dims(x_Test[0],axis=0), model.predict, num_features=10)
data_to_save = exp.as_list()

##############XAI################

explainer = lime_tabular.RecurrentTabularExplainer(x_Train,training_labels = yy_Train, feature_names=("SERS"), discretize_continuous=False, feature_selection= 'auto', class_names=['class1', 'class2'])
data_save = []


data_save = []
for i in range(19600):
    exp = explainer.explain_instance(np.expand_dims(x_Test[i],axis=0), model.predict, num_features=10)
    data_to_save = exp.as_list()
    data_save.append(data_to_save)
    
with open("D:/KSM/TRD/Result/240326/Second_Test_XAI.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data_save)






