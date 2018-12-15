# AttriGuard: A Practical Defense Against Attribute Inference Attacks via Adversarial Machine Learning

This repository contains dataset and code for "AttriGuard: A Practical Defense Against Attribute Inference Attacks via Adversarial Machine Learning". 

Required python tool (under Ubuntu 16.04): Python 2.7, Keras (2.2.2), TensorFlow 1.9.0 (as we compute gradient, different backend may have different implementation, you maybe need to check the backend), cvxpy (0.4.9), numpy, argparse (1.1). GPU support (you can also comment related code if not using GPU). 

This is an optimized version which is more efficient than our original version.  

We also use a better setting (we changed setting like batch size, learning rate, loss function and epochs) of neural network classifiers than we originally used in our paper for both attacker and defender to better show the effectiveness of the proposed method. 

# Dataset description: 

train_user.txt contains the training data, each row represents apps rated by an user. For example, in 3:1.0, 3 denotes app id and 1.0 represents rating score (we normalize rating scores from 0-5 to 0-1). 

train_label.txt contains user city id. For example, first row is 5 represents user lives/lived in city 5. There are 25 cities (0-24). 

test_user.txt and test_label.txt is similar to training dataset. 

# Code usage: 
input_data.py is used to read training data and testing data. 

keras_model.py contains defense model and attack model. 

nn_attack.py nn_defense.py are used to train attack and defense models, respectively. 

phaseI.py is the code for Phase I of AttriGuard. Note here we use "Modify_Add" policy. 

phaseII.py is the code for Phase II of AttriGuard. Note we use cvxpy (version 0.4.9, http://www.cvxpy.org/) to implement our phase II. 

You can directly run shell run.sh (chmod +x run.sh run_phaseI.sh) after installing python tools. It will automatically run the pipeline. 

We also run the code and obtain the following result (defender uses NN, attacker also uses NN): (similar to Figure 4 in AttriGuard paper). We can provide pre-trained model if needed. 

Utility Budget List: [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

Inference Accuracy List: [0.41, 0.37, 0.30, 0.18, 0.12, 0.09, 0.08]

This code is implemented by Jinyuan Jia. If you have any question, please feel free to send email to jinyuanjia02@gmail.com. 

# Citation
If you use this code or dataset, please cite following paper: 
<a href="https://arxiv.org/pdf/1805.04810.pdf">AttriGuard</a>
```
@inproceedings{jia2018attriguard,
  title={{AttriGuard}: A Practical Defense Against Attribute Inference Attacks via Adversarial Machine Learning},
  author={Jinyuan Jia and Neil Zhenqiang Gong},
  booktitle={USENIX Security Symposium},
  year={2018}
}
```