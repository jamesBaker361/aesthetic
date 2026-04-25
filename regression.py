import numpy as np
import argparse
import os
import csv
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class ElasticRegression:
    def __init__(self, learning_rate, iterations, l1_penalty, l2_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = np.zeros(self.n)
        for j in range(self.n):
            l1_grad = self.l1_penalty if self.W[j] > 0 else -self.l1_penalty
            dW[j] = (
                -2 * (self.X[:, j]).dot(self.Y - Y_pred) +
                l1_grad + 2 * self.l2_penalty * self.W[j]
            ) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
        return self

    def predict(self, X):
        return X.dot(self.W) + self.b

parser=argparse.ArgumentParser()
parser.add_argument("--y_column",type=str,default="aesthetic") #column 0 = aesthetic column = 1 = p(unsafe)
parser.add_argument("--block",type=str,default="down_blocks.2.attentions.1")
parser.add_argument("--limit",type=int,default=-1)

info_path="laion/info.csv"
sparse_dir="sparse_embeddings"
dest_dir="statistics"

os.makedirs(dest_dir,exist_ok=True)

if __name__=="__main__":
    args=parser.parse_args()
    dependent=[]
    independent=[]
    with open(info_path,"r") as file:
        for l,line in enumerate(tqdm(file.readlines())):
            if l==args.limit:
                break
            [imgpath,aesthetic,punsafe]=line.strip().split(",")
            imgpath=imgpath.split("/")[1]
            aesthetic=float(aesthetic)
            punsafe=float(punsafe)
            target={
                "aesthetic":aesthetic,
                "punsafe":punsafe
            }[args.y_column]
            if l<10:
                print(target)
            npz_file=os.path.join(sparse_dir,imgpath+".npz")
            if os.path.exists(npz_file):
                features=np.load(npz_file)[args.block]
                n=features.shape[0]
                if l<10:
                    print(features.shape)
                
                features=[f for f in features if np.isfinite(f).all()]
                independent.extend([f for f in features])
                dependent.extend([target for _ in range(len(features))])
            elif l<10:
                print(npz_file,"doesnt exists")
            
    print(" len samples",len(dependent))
    
    independent=np.array(independent)
    dependent=np.array(dependent)

    indep_mean = independent.mean(axis=0)
    indep_std = independent.std(axis=0)
    indep_std[indep_std == 0] = 1
    independent = (independent - indep_mean) / indep_std
    independent = np.hstack([independent, np.ones((independent.shape[0], 1))])

    dep_mean = dependent.mean()
    dep_std = dependent.std()
    dependent = (dependent - dep_mean) / dep_std

    split=int(len(dependent)*0.05)
    indep_test=independent[:split]
    indep_train=independent[split:]
    dep_test=dependent[:split]
    dep_train=dependent[split:]
    
    
    for var,name in zip([indep_train, indep_test, dep_train, dep_test,independent,dependent],
                        ["indep_train", "indep_test", "dep_train", "dep_test","independent","dependent"]):
        print(name,var.shape)
    print("indep test")
    t0=time.time()
    x,residuals,rank,s=np.linalg.lstsq(indep_train,dep_train,rcond=None)
    print(f"lstsq: {time.time()-t0:.2f}s")
    t0=time.time()
    covariance=np.corrcoef(independent)
    print(f"covariance: {time.time()-t0:.2f}s")
    
    pred=x @ indep_test
    print("linear regression")
    mse = mean_squared_error(dep_test, pred)
    print("\tmse ",mse)
    rmse=np.sqrt(mse)
    print("\trmse ",rmse)
    r2 = r2_score(dep_test, pred)
    print("\tR2 Score:", r2)
    
    #elastic regression
    model = ElasticRegression(
        iterations=10000,
        learning_rate=0.01,
        l1_penalty=500,
        l2_penalty=1
    )
    model.fit(indep_train,dep_train)
    print("elastic regression")
    pred=model.predict(indep_test)
    mse = mean_squared_error(dep_test, pred)
    print("\tmse ",mse)
    rmse=np.sqrt(mse)
    print("\trmse ",rmse)
    r2 = r2_score(dep_test, pred)
    print("\tR2 Score:", r2)
    
    np.savez(os.path.join(dest_dir,args.block),weights_linear=x,covar=covariance)
    