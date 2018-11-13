import numpy as np

user_each_city=np.zeros([25],dtype=np.float)
train_label=np.loadtxt("./data/train_label.txt",dtype=np.int)  

for i in np.arange(train_label.shape[0]):
    user_each_city[train_label[i]]+=1.0

for i in np.arange(25):
    user_each_city[i]=user_each_city[i]/train_label.shape[0]
print(user_each_city)

np.savetxt("./data/prior.txt",user_each_city)