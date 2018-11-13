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


t=Data_Class()

t.input_train_app()
t.input_train_label()
t.input_test_app()
t.input_test_label()

learning_rate=0.05
batch_size=100
epochs=50
save_model=True

input_shape=t.train_app.shape[1:]
x_train=t.train_app
y_train=t.train_label
x_test=t.test_app
y_test=t.test_label


model=keras_model.cnn_model_defense(input_shape=input_shape)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=learning_rate),metrics=['accuracy'])
model.summary()

benign_index_array=np.arange(x_train.shape[0])
batch_num=np.int(np.ceil(x_train.shape[0]/batch_size))
for i in np.arange(epochs):
    print("epoch {}".format(i))
    for j in np.arange(batch_num):       
        x_batch=x_train[benign_index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,x_train.shape[0])],:]
        y_batch=y_train[benign_index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,x_train.shape[0])],:]
        model.train_on_batch(x_batch,y_batch)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss: {}".format(scores[0]))
    print("Test accuracy: {}".format(scores[1]))
if save_model:
    weights=model.get_weights()
    np.savez("./models/cnn_model_defense_weights.npz",x=weights)
