from input_data import *
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import keras_model
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("-tl",type=int)
args=parser.parse_args()

config = tf.ConfigProto()
#you may need to comment following code if not using gpu
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.visible_device_list = "0"

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

t=Data_Class()

t.input_train_app()
t.input_train_label()
t.input_test_app()
t.input_test_label()

learning_rate=0.05
batch_size=1000
epochs=50
save_model=True

input_shape=t.train_app.shape[1:]
x_train=t.train_app
y_train=t.train_label
x_test=t.test_app
y_test=t.test_label


npzdata=np.load("./models/cnn_model_defense_weights.npz")
weights=npzdata['x']

model=keras_model.cnn_model_defense(input_shape=input_shape)

for i in np.arange(len(model.layers)):
        model.layers[i].trainable=False
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=learning_rate),metrics=['accuracy'])

model.set_weights(weights)

model.summary()
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss: {}".format(scores[0]))
print("Test accuracy: {}".format(scores[1]))
target_label=int(args.tl)
print("Target label: {}".format(target_label))
output=model.layers[-2].output

gradient_targetlabel=K.gradients(output[:,target_label],model.input)

maxiter=100
result_array=np.copy(x_test)
noise_add=0.0
for i in np.arange(x_test.shape[0]):
    if i%100==0:
        print(i)
    sample=result_array[i,:].reshape(1,10000)
    predict_label=np.argmax(model.predict(sample),axis=1)[0]
    j=0
    while predict_label!=target_label and j<maxiter:
        gradient_value=sess.run(gradient_targetlabel,feed_dict={model.input:sample})[0][0]
        max_index=np.argmax((1.0-sample)*gradient_value)
        min_index=np.argmax(-1.0*sample*gradient_value)      
        if (1.0-sample[0,max_index])*gradient_value[max_index]>=(-1.0)*sample[0,min_index]*gradient_value[min_index]:
            sample[0,max_index]=1.0
        else:
            sample[0,min_index]=0.0
        predict_label=np.argmax(model.predict(sample),axis=1)[0]
        j+=1
    result_array[i,:]=sample[0,:]
    noise_add+=j
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss: {}".format(scores[0]))
print("Test accuracy: {}".format(scores[1]))
scores = model.predict(result_array)
predict_value=np.argmax(scores,axis=1)
print("average added noise: {}".format(noise_add/x_test.shape[0]))

savefilepath="./data/adversarial_sample/"+str(target_label)+".txt"
output_train_file=open(savefilepath,'w')
for i in range(result_array.shape[0]):
    line=''
    nonzero=np.nonzero(result_array[i,:])[0]
    for j in range(nonzero.shape[0]):
        if result_array[i,nonzero[j]]!=0:
            line+=str(nonzero[j])
            line+=':'
            line+='{0:.1f}'.format(result_array[i,nonzero[j]])
            line+=' '
    line=line[0:-1]
    output_train_file.write(line)
    output_train_file.write('\n')
output_train_file.close()


