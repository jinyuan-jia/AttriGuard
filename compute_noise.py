from input_data import *
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import keras_model

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

t=Data_Class()
t.input_test_app()


noise_num=np.zeros([number_of_test,number_of_city],dtype=np.int)
for i in np.arange(number_of_city):
    filepath="./data/adversarial_sample/"+str(i)+".txt"
    adversarial_sample=input_adversarial_example(filepath)
    overall_noise=0.0
    noise_number=0
    for j in np.arange(t.test_app.shape[0]):
        difference=t.test_app[j,:]-adversarial_sample[j,:]
        nonzero=np.nonzero(difference)[0]
        noise_number=nonzero.shape[0]
        noise_num[j,i]=noise_number
        overall_noise+=noise_number
    print("noise added:{}".format(overall_noise/t.test_app.shape[0]))
np.savetxt("./data/adversarial_sample/noise.txt",noise_num,fmt="%i")


