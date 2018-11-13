import numpy as np
import cvxpy as cp

noise_array=np.loadtxt("./data/adversarial_sample/noise.txt",dtype=np.int)
prob_array=np.loadtxt("./data/prior.txt",dtype=np.float)
transfer_array=np.loadtxt("./data/attack_prediction/nn.txt",dtype=np.int)
test_label=np.loadtxt("./data/test_label.txt",dtype=np.int)  

epinsonarray=[0.3,0.5,1.0,2.0,3.0,4.0,5.0]
Dimension=25
Prior=prob_array
Q=cp.Variable(Dimension)
objective=cp.sum_entries(cp.kl_div(Prior,Q))
obj=cp.Minimize(objective)
probcount=0.
probcountlist=[]
for epinslon in epinsonarray:
    probcount=0.
    print("Utility budget: {}".format(epinslon))
    for ii in range(test_label.shape[0]):
        if ii%100==0:
            print(ii)
        true_label=test_label[ii]
        constraints=[Q>1e-10,cp.sum_entries(Q)==1,noise_array[ii,:]*Q<epinslon]
        prob=cp.Problem(obj,constraints)
        result=prob.solve(solver=cp.CVXOPT,max_iters=1000,abstol=1e-4,reltol=1e-4,feastol=1e-4)
        for i in range(Dimension):
           if abs(transfer_array[ii,i]-true_label)<0.01:
               probcount+=Q.value[i,0]  
    print('Precision: {}'.format(probcount/test_label.shape[0]))
    probcountlist.append(probcount/(test_label.shape[0]+0.0))
print("Utility budget list: {}".format(epinsonarray))
print("Precision List: {}".format(probcountlist)) 