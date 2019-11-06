import numpy as np
import pandas as pd
import os
import sys

train_data=pd.read_csv(sys.argv[1],header=None)
test_data = pd.read_csv(sys.argv[2],header=None)
def normalise(data):
    data = data/255
    return data.iloc[:train_data.shape[0],:] , data.iloc[train_data.shape[0]:,:]

data = pd.concat([train_data,test_data],axis=0)
X_train , X_test = normalise(data.iloc[:,:-1])
Y_train = data.iloc[:train_data.shape[0],-1]
print(X_test.shape)
train_data = pd.concat([X_train,Y_train],axis=1)

train_dic = {}
for i,c in enumerate(train_data.iloc[:,-1]):
    if(c not in train_dic.keys()):
        train_dic[c]=[]
    train_dic[c].append(train_data.iloc[i,:].values)    


def solve(iteration,lambda_value,data,indexes):
    w = np.zeros(784)
    b=0
    for i in range(1,iteration+1):
        eta = 1/(lambda_value*i)
        # k = np.random.randint(0,data.shape[0])
        y = data[indexes[i-1],-1]
        x = data[indexes[i-1],:-1]
        val = y*(np.dot(w,x)+b)
        if(val<1):
            w = (1-eta*lambda_value)*w + eta*y*x
            b = b + eta*y
        else:
            w = (1-eta*lambda_value)*w
    return w , b       
def predict(X):
    Y_pred = []
    for i in range(10):
        for j in range(i+1,10):
            y_pred = np.dot(X , w[i][j].reshape(784,1))+b[i][j]
            y_pred = np.where(y_pred<=0,i,j)
            Y_pred.append(y_pred)
    y = []
    for i in range(X.shape[0]):
        print(i,end="\r",flush=True)
        y.append(np.argmax(np.bincount(np.array(Y_pred)[:,i].reshape(45))))
    return np.array(y) 
itr = 40000
lm = 0.32
w = [[None for i in range(45)] for i in range(45)]
b = [[None  for i in range(45)] for i in range(45)]
for i in range(10):
    for j in range(i+1,10):
        print(i,j)
        d = np.array(train_dic[i]+train_dic[j])
        d[:,-1] = np.where(d[:,-1]==i,-1,1)
        np.random.seed(0)
        np.random.shuffle(d)
        np.random.seed(0)
        indexes = np.random.randint(0,d.shape[0],size=itr).tolist()
        w[i][j] , b[i][j] = solve(itr,lm,d[:,:],indexes)
Y_pred = predict(X_test)
print(Y_pred.shape)
np.savetxt(sys.argv[3],Y_pred)