import numpy as np
import argparse
import os
import csv
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from experiment_helpers.argprint import print_args
from sklearn.linear_model import Ridge,LinearRegression,ElasticNet,Lasso


parser=argparse.ArgumentParser()
parser.add_argument("--y_column",type=str,default="aesthetic") #column 0 = aesthetic column = 1 = p(unsafe)
parser.add_argument("--block",type=str,default="down_blocks.2.attentions.1")
parser.add_argument("--limit",type=int,default=-1)

info_path="laion/info.csv"
sparse_dir="sparse_embeddings"
dest_dir="statistics"

os.makedirs(dest_dir,exist_ok=True)

if __name__=="__main__":
    print_args(parser)
    args=parser.parse_args()
    print(args)
    indep_chunks=[]
    dependent=[]
    with open(info_path,"r") as file:
        for l,line in enumerate(tqdm(file)):
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
                if l<10:
                    print(features.shape)
                mask=np.isfinite(features).all(axis=1)
                features=features[mask]
                if len(features):
                    indep_chunks.append(features)
                    dependent.extend([target]*len(features))
            elif l<10:
                print(npz_file,"doesnt exists")

    print(" len samples",len(dependent))

    independent=np.vstack(indep_chunks)
    del indep_chunks
    dependent=np.array(dependent)

    indep_mean = independent.mean(axis=0)
    indep_std = independent.std(axis=0)
    indep_std[indep_std == 0] = 1
    independent = (independent - indep_mean) / indep_std
    #independent = np.hstack([independent, np.ones((independent.shape[0], 1))]) bias doesn't rlly matter?
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
        
    npz_dict={}
    for solver_class,name in zip(
            [LinearRegression,ElasticNet,Ridge,Lasso],
            ["LinearRegression","ElasticNet","Ridge","Lasso"]):
        model=solver_class()
        t0=time.time()
        model.fit(indep_train,dep_train)
        for key,value in model.get_params().items():
            npz_dict[f"{name}_{key}"]=value
        print(f"{name} {time.time()-t0:.2f}s")
        
        

    
    
    t0=time.time()
    covariance=np.corrcoef(independent,rowvar=False)
    print(f"covariance: {time.time()-t0:.2f}s")
    
    np.savez(os.path.join(dest_dir,args.block,args.y_column),
             covar=covariance,
             indep_mean=indep_mean,
             dep_mean=dep_mean,
             indep_std=indep_std,
             dep_std=dep_std,**npz_dict)
    