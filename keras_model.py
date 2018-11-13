from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import Dense,Input,Dropout,Activation
import tensorflow as tf

def cnn_model_defense(input_shape,num_classes=25):
    inputs=Input(shape=input_shape)
    x=Dense(50000,activation='relu')(inputs)
    x=Dropout(0.7)(x)
    x=Dense(25)(x)
    output=Activation('softmax')(x)
    model=Model(inputs=inputs,outputs=output)
    return model

def cnn_model_attack(input_shape,num_classes=25):
    inputs=Input(shape=input_shape)
    x=Dense(30000,activation='relu')(inputs)
    x=Dropout(0.7)(x)
    output=Dense(25,activation='softmax')(x)
    model=Model(inputs=inputs,outputs=output)
    return model





