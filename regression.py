import numpy as np
import argparse
import os
import csv
import time
from tqdm import tqdm


parser=argparse.ArgumentParser()
parser.add_argument("--y_column",type=str,default="aesthetic") #column 0 = aesthetic column = 1 = p(unsafe)
parser.add_argument("--block",type=str,default="down_blocks.2.attentions.1")
parser.add_argument("--limit",type=int,default=-1)

info_path="laion/info.csv"
sparse_dir="sparse_embeddings"
dest_dir="statistics"

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
                dependent.extend([target for _ in range(n)])
                features=[f for f in features if np.isfinite(f).all()]
                independent.extend([f for f in features])
            elif l<10:
                print(npz_file,"doesnt exists")
            
    print(" len samples",len(dependent))
    
    independent=np.array(independent)
    dependent=np.array(dependent)
    print("dependnent",dependent.shape)
    print("indpednent ",independent.shape)
    t0=time.time()
    x,residuals,rank,s=np.linalg.lstsq(independent,dependent,rcond=None)
    print(f"lstsq: {time.time()-t0:.2f}s")
    t0=time.time()
    covariance=np.corrcoef(independent)
    print(f"covariance: {time.time()-t0:.2f}s")
    np.savez(os.path.join(dest_dir,args.block),weights=x,covar=covariance)
    