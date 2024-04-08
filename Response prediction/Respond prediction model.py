# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 19:58:10 2023

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


""" Set Random Seed for Uniform Performance """
tf.random.set_seed(777)
np.random.seed(777)

if 'Dataset_Test' in locals():
    print('Loading Dataset Done')
else: 
        Dataset_Test = np.loadtxt('C:/SMinKIM/TRD monitoring/data/231215/with label/Total/Final_Test.csv', dtype=np.float32)
        Dataset_Train = np.loadtxt('C:/SMinKIM/TRD monitoring/data/231215/with label/Total/Second_Train_CV5.csv', dtype=np.float32)
        Dataset_Val = np.loadtxt('C:/SMinKIM/TRD monitoring/data/231215/with label/Total/Second_Val_CV5.csv', dtype=np.float32)


x_Train = Dataset_Train[:,:800]
y_Train = Dataset_Train[:,800]
x_Val = Dataset_Val[:,:800]
y_Val = Dataset_Val[:,800]
x_Test = Dataset_Test[:,:800]
y_Test = Dataset_Test[:,800]
    
    
shuffle_indices = np.random.permutation(np.arange((len(y_Train))))
x_Train = copy.copy(x_Train[shuffle_indices,:])
y_Train = copy.copy(y_Train[shuffle_indices])
    
  # Reshape Dataset (For matching with input size)
x_Train = x_Train.reshape((len(x_Train), np.prod(x_Train.shape[1:]),1))
x_Val = x_Val.reshape((len(x_Val), np.prod(x_Val.shape[1:]),1))
x_Test = x_Test.reshape((len(x_Test), np.prod(x_Test.shape[1:]),1))


momentum = 0.9
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

EPOCH = 200

opi = 1    

BATCHSIZE = 406
LEARNINGRATE = 0.003 
DECAY = 0.4
DO = 0.2
REG = 0.0001
parameters = [BATCHSIZE, LEARNINGRATE, DECAY, DO, REG]

# Building model
Input = tf.keras.Input(shape=(800, 1))
        
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
x1 = Dense(512, activation='relu', kernel_regularizer=l2(REG))(x1)
x1 = Dense(512, name='fcl2', kernel_regularizer=l2(REG))(x1)
x1 = Activation("relu")(x1)
x1 = Dropout(DO)(x1)
        
x1 = Dense(256, activation='relu', kernel_regularizer=l2(REG))(x1)
x1 = Dense(256,  kernel_regularizer=l2(REG))(x1)
x1 = Activation("relu")(x1)
       
Output = Dense(1, activation='sigmoid')(x1)
        
model = tf.keras.Model(Input, Output)
model.summary()
        
model.compile(optimizer=RMSprop(lr=LEARNINGRATE, decay=DECAY),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        
# Training
history = model.fit(x_Train, y_Train,
                    epochs=EPOCH,
                    batch_size=BATCHSIZE,
                    shuffle=True,
                    validation_data = (x_Val, y_Val))
    
    

# test set accuracy
results = model.evaluate(x_Test, y_Test)
    

today=datetime.datetime.now()
day = str(today.year)[2:4] +str(today.month+100)[1:3] + str(today.day+100)[1:3]
time = '_' + str(today.hour+100)[1:3] +'_'+ str(today.minute+100)[1:3] + '_' + str(today.second+100)[1:3]
        
save_path_time = 'C:/SMinKIM/TRD monitoring/Result' + '/' + day + '/'


Test_Res = Dataset_Test[0:8000, :800] # Before excluding misdiagnosed samples from the first model
Test_Non = Dataset_Test[8000:, :800]
    
Test_Res = Test_Res.reshape((len(Test_Res), np.prod(Test_Res.shape[1:]),1))
Test_Non = Test_Non.reshape((len(Test_Non), np.prod(Test_Non.shape[1:]),1))
    
output_Res = model.predict(Test_Res)
output_Non = model.predict(Test_Non)
    
set_Res = np.split(output_Res, 20) # Before excluding misdiagnosed samples from the first model
set_Non = np.split(output_Non, 29) # Before excluding misdiagnosed samples from the first model

bar_Res = np.zeros((len(set_Res),))
bar_Non = np.zeros((len(set_Non),))
    
for i in range(len(set_Res)):
    a = set_Res[i]
    bar_Res[i] = np.mean(a[:,:])
for i in range(len(set_Non)):
    a = set_Non[i]
    bar_Non[i] = np.mean(a[:,:])
        
y_score = np.hstack((bar_Res, bar_Non))
y_true = np.hstack((np.zeros(len(bar_Res,)), np.ones(len(bar_Non,))))
        
def rocvis(true , prob , label ) :
    if type(true[0]) == str :
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true = le.fit_transform(true)
    else :
        pass
        
fpr, tpr, thresholds = roc_curve(y_true, y_score,1) 
AUC = auc(fpr, tpr)
plt.title("Test_AUC_" + str(AUC) , fontsize= 15)
plt.plot(fpr, tpr, marker='.' )
fig1 = plt.gcf() 
plt.show()
fig1.savefig(save_path_time  + '/'+
            str(AUC) + '%Test_AUC_' +str(opi) +'.png')  
   # bar graph
label = y_true
    
index1 = np.arange(0,20) # Before excluding misdiagnosed samples from the first model
index2 = np.arange(22,22+29) # Before excluding misdiagnosed samples from the first model
#plt.bar(index,y_score)?i'm
bar_width = 0.5
p1 = plt.bar(index1, bar_Res, bar_width, color='k', label='HC')
p2 = plt.bar(index2 , bar_Non, bar_width, color='r', label='DP')
plt.title("Test_Bar" + str(AUC) , fontsize= 15)
fig2 = plt.gcf() 
plt.show()
fig2.savefig(save_path_time  + '/'+
            str(AUC) + '%Test_Bar_' +str(opi) +'.png')  
    
    
    #############################################################################################
        
fig = plt.figure(figsize=(10, 7))
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('BATCHSIZE: ' + str(parameters[0]) +
          '  //  LEARNINGRATE: ' + str(parameters[1]) +
          '  //  DECAY: ' + str(parameters[2]) +
          '  //  DO: ' + str(parameters[3]) +
          '  //  REG: ' + str(parameters[4]))

plt.ylabel('accuracy')
plt.legend(['train', 'val'], loc='upper left')
      # Loss monitoring
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
fig.savefig(save_path_time  + '/'+
            str(int(round(100*history.history['val_accuracy'][-1],0))) + '%_' + str(results[1]) +'.png')  
    
np.savetxt(save_path_time + '/'+"Tra_loss_"+ str(opi) +".csv", history.history['loss'], delimiter = ",") 
np.savetxt(save_path_time + '/'+"Val_loss_"+ str(opi) +".csv", history.history['val_loss'], delimiter = ",") 
np.savetxt(save_path_time + '/'+"Tra_acc_"+ str(opi) +".csv", history.history['accuracy'], delimiter = ",") 
np.savetxt(save_path_time + '/'+"Val_acc_"+ str(opi) +".csv", history.history['val_accuracy'], delimiter = ",") 
np.savetxt(save_path_time + '/'+"Res_output.txt", output_Res, fmt='%.18e', delimiter=',')
np.savetxt(save_path_time + '/'+"Non_output.txt", output_Non,  fmt='%.18e', delimiter=',')

model_json = model.to_json()
with open(save_path_time + '/' + str(opi) + '.json', "w") as json_file:
    json_file.write(model_json)
   # Save weight with h5 format
    model.save_weights(save_path_time + '/' + 'Saved_Model_Weight_'+ str(opi) +'.h5')
    print("Saved model to disk")
            
            
     
tf.keras.backend.clear_session()

