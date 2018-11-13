import numpy as np


train_app_filepath="./data/train_user.txt"
train_label_filepath="./data/train_label.txt"
test_app_filepath="./data/test_user.txt"
test_label_filepath="./data/test_label.txt"

class Data_Class:
    def __init__(self):
        self.number_of_train=14607
        self.number_of_app=10000
        self.number_of_label=25
        self.number_of_test=1631
#####################################################
    def input_train_app(self):
        self.train_app=np.zeros([self.number_of_train,self.number_of_app],dtype=np.float32)
        input_train_file=open(train_app_filepath,'rb')
        i=0
        for line in input_train_file:
            line=line[0:-1]
            for appnum in line.split(' '):
                appnumdetail=appnum.split(':')
                self.train_app[i,int(appnumdetail[0])]=float(appnumdetail[1])
            i+=1
        input_train_file.close()
#####################################################
    def input_test_app(self):
        self.test_app=np.zeros([self.number_of_test,self.number_of_app],dtype=np.float32)
        input_test_file=open(test_app_filepath,'rb')
        i=0
        for line in input_test_file:
            line=line[0:-1]
            for appnum in line.split(' '):
                appnumdetail=appnum.split(':')
                self.test_app[i,int(appnumdetail[0])]=float(appnumdetail[1])
            i+=1
        input_test_file.close()
#####################################################
    def input_train_label(self):
        self.train_label=np.full([self.number_of_train,self.number_of_label],0,dtype=np.float32)
        input_train_label=np.loadtxt(train_label_filepath,dtype=np.int)
        for i in np.arange(self.train_label.shape[0]):
            self.train_label[i,input_train_label[i]]=1
#####################################################
    def input_test_label(self):
        self.test_label=np.full([self.number_of_test,self.number_of_label],0,dtype=np.float32)
        input_test_label=np.loadtxt(test_label_filepath,dtype=np.int)
        for i in np.arange(self.test_label.shape[0]):
            self.test_label[i,input_test_label[i]]=1
#####################################################




