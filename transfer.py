from input_data import *
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import keras_model

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
number_of_test=1631
number_of_app=10000
number_of_city=25
def input_adversarial_example(adversarial_example_filepath):
    global number_of_test,number_of_app
    adversarial_example=np.zeros([number_of_test,number_of_app],dtype=np.float32)
    input_adversarial_file=open(adversarial_example_filepath,'rb')
    i=0
    for line in input_adversarial_file:
        line=line[0:-1]
        for appnum in line.split(' '):
            appnumdetail=appnum.split(':')
            adversarial_example[i,int(appnumdetail[0])]=float(appnumdetail[1])
        i+=1
    input_adversarial_file.close()
    return adversarial_example
learning_rate=0.05
input_shape=(10000,)
npzdata=np.load("./models/cnn_model_attack_weights.npz")
weights=npzdata['x']
model=keras_model.cnn_model_attack(input_shape=input_shape)
for i in np.arange(len(model.layers)):
        model.layers[i].trainable=False
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=learning_rate),metrics=['accuracy'])
model.set_weights(weights)
model.summary()

attack_prediction_value=np.zeros([number_of_test,number_of_city],dtype=np.int)
for i in np.arange(number_of_city):
    filepath="./data/adversarial_sample/"+str(i)+".txt"
    adversarial_sample=input_adversarial_example(filepath)
    scores = model.predict(adversarial_sample)
    predict_value=np.argmax(scores,axis=1)
    attack_prediction_value[:,i]=predict_value[:]
np.savetxt("./data/attack_prediction/nn.txt",attack_prediction_value,fmt="%i")


