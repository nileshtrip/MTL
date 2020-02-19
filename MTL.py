#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import autograd.numpy as np
from autograd import grad
import numpy.linalg as la
import scipy.optimize
import random
import math

import ray
import time
import datetime
import mkl
import os

import pickle


# In[ ]:


ray.init(num_cpus=48, redis_password="123456")


# In[ ]:


def sin_angle(B1, B2):
    
    d, r = B1.shape
    svs = la.svd(B1.T @ B2)[1]
    cos_theta = min(svs)
    sin_theta = math.pow(1-cos_theta**2, 0.5)
    
    return sin_theta


# In[ ]:


def eigs(M):
    
    eigenValues, eigenVectors = la.eig(M)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    return eigenValues, eigenVectors


# In[ ]:


def get_col_space(B, r):
    
    u, _, _ = la.svd(B)
    
    return u[:, 0:r]


# In[ ]:


def gen_train_model(d, r, T, train_n):
    
    u, s, v = la.svd(np.random.normal(size=(d, r)))
    B = u[:, :r]
    
    train_alphas = [np.random.normal(size=r, scale=1/math.sqrt(r)) for i in range(T)]
    train_data=[]
    for i in range(T):
        X=np.random.normal(size=(train_n, d))
        y = X @ B @ train_alphas[i] + np.random.normal(size=train_n)
        train_data.append((X, y))
        
    return train_data, B, train_alphas


# In[ ]:


def gen_test_model(d, r, B, test_n):
    
    alpha = np.random.normal(size=r, scale=1/math.sqrt(r))

    X=np.random.normal(size=(test_n, d))
    y = X @ B @ alpha + np.random.normal(size=test_n)
        
    return (X, y), alpha


# In[ ]:


def MoM(train_data):
    
    T = len(train_data)
    d = train_data[0][0].shape[1]
    
    total_n=0
    M = np.zeros(shape=(d, d))
    for i in range(T):
        data = train_data[i]
        X, y = data
        num = y.shape[0]
        total_n += num
        scaled_X = (X.T * y).T
        M += (scaled_X).T @ scaled_X
    M = 1/float(total_n) * M
    
    return M


# In[ ]:


def rPCA(M, r):
    
    eigVals, eigVecs = eigs(M)
    
    return eigVecs[:, :r], eigVecs[:, r:]


# In[ ]:


def change_shape(w, d, r, T):
    
    b=w[:d*r]
    v=w[d*r:]
    
    B = np.reshape(b, (d,r))
    V = np.reshape(v, (T,r))
    
    return B, V


# In[ ]:


def MS_Loss(weights, train_data, d, r, m):
    

    T = len(train_data)
    
    b=weights[:d*r]
    v=weights[d*r:]
    
    B = np.reshape(b, (d,r))
    V = np.reshape(v, (T,r))
    
    loss=0
    for t in range(T):
        X, y = train_data[t]
        loss += 1/(2*m)*np.linalg.norm(y-X @ B @ V[t, :])**2
       
    loss += 1/8*np.linalg.norm(B.T @ B - V.T @ V, "fro")**2
    
    return loss


# In[ ]:


def LR_Loss(weights, test_data):
    
    X = test_data[0]
    y = test_data[1]
    
    n = y.shape[0]
    loss = 1/(2*n)*np.linalg.norm(y-X @ weights)**2
    
    return loss


# In[ ]:


def MetaLR_w_MOM(train_data, r, test_data):
    
    T = len(train_data)
    d = train_data[0][0].shape[1]
    
    M_est = MoM(train_data)
    B1, B2 = rPCA(M_est, r)
    
    X,y = test_data
    X_low = X @ B1
    alpha_LR = LR((X_low, y))
    beta_LR = B1 @ alpha_LR
    
    return B1, beta_LR


# In[ ]:


def MetaLR_w_FO(train_data, r, test_data):
    
    T = len(train_data)
    n, d = train_data[0][0].shape
    m = T*n
    
    ms_gradients = grad(MS_Loss)
    
    B_init = np.random.normal(size=(d,r)).flatten()
    V_init = np.random.normal(size=(T,r)).flatten()
    w = np.concatenate((B_init, V_init))
    
    res_ms = scipy.optimize.minimize(MS_Loss, w, jac=ms_gradients, method='L-BFGS-B', args=(train_data, d, r, m), options = {'maxiter' : 1000})   
    B_gd, V_gd = change_shape(res_ms.x, d, r, T)
    B1 = get_col_space(B_gd, r)
    
    X, y = test_data
    X_low = X @ B1
    test_data_new = [X_low, y]
    
    test_gradients = grad(LR_Loss)
    
    w = np.random.normal(size=r).flatten()
    res_test = scipy.optimize.minimize(LR_Loss, w, jac=test_gradients, method='L-BFGS-B', args=(test_data_new), options = {'maxiter' : 1000})  
    alpha_LR = res_test.x
    
    beta_LR = B1 @ alpha_LR
    
    return B1, beta_LR


# In[ ]:


def LR(test_data):
    
    X, y = test_data
    beta_LR = la.pinv((X.T @ X)) @ X.T @ y
    
    return beta_LR


# In[ ]:


@ray.remote
def run_expt(d, r, T, train_n, test_n, seed):
    
    mkl.set_num_threads(4)
    np.random.seed(seed)
    train_data, B, train_alphas = gen_train_model(d=d, r=r, T=T, train_n=train_n)
    test_data, alpha_test = gen_test_model(d, r, B, test_n)
    
    B_meta_mom, beta_meta_LR_mom = MetaLR_w_MOM(train_data, r, test_data)
    sin_theta_mom = sin_angle(B_meta_mom, B)
    
    B_meta_fo, beta_meta_LR_fo = MetaLR_w_FO(train_data, r, test_data)
    sin_theta_fo = sin_angle(B_meta_fo, B)
    
    beta_LR = LR(test_data)
    beta_true = B @ alpha_test

    return np.linalg.norm(beta_meta_LR_mom-beta_true), np.linalg.norm(beta_meta_LR_fo-beta_true), np.linalg.norm(beta_LR-beta_true), sin_theta_mom, sin_theta_fo 


# In[ ]:


def run_parallel_expt(d, r, T, train_n, test_n, reps):
    
    meta_LR_errs=[]
    meta_RR_errs=[]
    LR_errs=[]
    ridge_errs=[]
    seeds = [i for i in range(reps)]

    data = ray.get([run_expt.remote(d, r, T, train_n, test_n, seeds[num]) for num in range(reps)])
    meta_LR_mom_errs, meta_LR_fo_errs, beta_LR_errs, sin_theta_mom, sin_theta_fo = zip(*data)
    
    return meta_LR_mom_errs, meta_LR_fo_errs, beta_LR_errs, sin_theta_mom, sin_theta_fo


# In[ ]:


#First Experiment with Large Test Set


# In[ ]:


d=100
r=5
train_n=5
test_n=2500
reps=30


# In[ ]:


T_list = [100, 200, 400, 800, 1600, 3200, 6400]


# In[ ]:


def collect_data_T(d, r, T_list, train_n, test_n, reps):
    
    metaLRmommus=[]
    metaLRmomstd=[]
    
    metaLRfomus=[]
    metaLRfostd=[]
    
    betaLRmus=[]
    betaLRstds=[]

    sinthetamommus=[]
    sinthetamomstd=[]
    
    sinthetafomus=[]
    sinthetafostd=[]
    
    for t in T_list:
        print(t)
        meta_LR_mom_errs, meta_LR_fo_errs, beta_LR_errs, sin_theta_mom, sin_theta_fo = run_parallel_expt(d, r, t, train_n, test_n, reps)

        metaLRmommus.append(np.mean(meta_LR_mom_errs))
        metaLRmomstd.append(np.std(meta_LR_mom_errs)) 

        metaLRfomus.append(np.mean(meta_LR_fo_errs))
        metaLRfostd.append(np.std(meta_LR_fo_errs)) 
        
        betaLRmus.append(np.mean(beta_LR_errs))
        betaLRstds.append(np.std(beta_LR_errs))

        sinthetamommus.append(np.mean(sin_theta_mom))
        sinthetamomstd.append(np.std(sin_theta_mom)) 

        sinthetafomus.append(np.mean(sin_theta_fo))
        sinthetafostd.append(np.std(sin_theta_fo)) 
        
    return (metaLRmommus, metaLRmomstd), (metaLRfomus, metaLRfostd), (betaLRmus, betaLRstds), (sinthetamommus, sinthetamomstd), (sinthetafomus, sinthetafostd)


# In[ ]:


metaLRmom, metaLRfo, betaLR, sinthetamom, sinthetafo = collect_data_T(d, r, T_list, train_n, test_n, reps)


# In[ ]:


save_data = {"metaLRmom" : metaLRmom, "metaLRfo" :  metaLRfo, "betaLR" : betaLR, "sinthetamom" : sinthetamom, "sinthetafo" : sinthetafo}


# In[ ]:


save_data["T_list"] =  T_list
save_data["d"] = d
save_data["r"] = r
save_data["train_n"] = train_n
save_data["test_n"] = test_n
save_data["reps"] = reps


# In[ ]:


params = "d="+str(d)+",r="+str(r)+",train_n="+str(train_n)+",test_n="+str(test_n)


# In[ ]:


file_name = "Meta,"+str(params)+".pickle"
folder_name = "Data"
file_path = os.path.join(folder_name, file_name)
pickle.dump(save_data, open(file_path, "wb"))


# In[ ]:


# Second Experiment with Small Train Set and Small Test Set


# In[ ]:


d=100
r=5
train_n=25
test_n=25
reps=30


# In[ ]:


metaLRmom, metaLRfo, betaLR, sinthetamom, sinthetafo = collect_data_T(d, r, T_list, train_n, test_n, reps)


# In[ ]:


save_data = {"metaLRmom" : metaLRmom, "metaLRfo" :  metaLRfo, "betaLR" : betaLR, "sinthetamom" : sinthetamom, "sinthetafo" : sinthetafo}


# In[ ]:


save_data["T_list"] =  T_list
save_data["d"] = d
save_data["r"] = r
save_data["train_n"] = train_n
save_data["test_n"] = test_n
save_data["reps"] = reps


# In[ ]:


params = "d="+str(d)+",r="+str(r)+",train_n="+str(train_n)+",test_n="+str(test_n)


# In[ ]:


file_name = "Meta,"+str(params)+".pickle"
folder_name = "Data"
file_path = os.path.join(folder_name, file_name)
pickle.dump(save_data, open(file_path, "wb"))


# In[ ]:


# Third Experiment Varying Training_n but with small number of Tasks


# In[ ]:


d=100
r=5
T=20
test_n=50
reps=30


# In[ ]:


train_n_list = [100, 200, 400, 800, 1600, 3200, 6400]


# In[ ]:


def collect_data_n(d, r, t, train_n_list, test_n, reps):
    
    metaLRmommus=[]
    metaLRmomstd=[]
    
    metaLRfomus=[]
    metaLRfostd=[]
    
    betaLRmus=[]
    betaLRstds=[]

    sinthetamommus=[]
    sinthetamomstd=[]
    
    sinthetafomus=[]
    sinthetafostd=[]
    
    for train_n in train_n_list:
        print(train_n)
        meta_LR_mom_errs, meta_LR_fo_errs, beta_LR_errs, sin_theta_mom, sin_theta_fo = run_parallel_expt(d, r, t, train_n, test_n, reps)

        metaLRmommus.append(np.mean(meta_LR_mom_errs))
        metaLRmomstd.append(np.std(meta_LR_mom_errs)) 

        metaLRfomus.append(np.mean(meta_LR_fo_errs))
        metaLRfostd.append(np.std(meta_LR_fo_errs)) 
        
        betaLRmus.append(np.mean(beta_LR_errs))
        betaLRstds.append(np.std(beta_LR_errs))

        sinthetamommus.append(np.mean(sin_theta_mom))
        sinthetamomstd.append(np.std(sin_theta_mom)) 

        sinthetafomus.append(np.mean(sin_theta_fo))
        sinthetafostd.append(np.std(sin_theta_fo)) 
        
    return (metaLRmommus, metaLRmomstd), (metaLRfomus, metaLRfostd), (betaLRmus, betaLRstds), (sinthetamommus, sinthetamomstd), (sinthetafomus, sinthetafostd)


# In[ ]:


metaLRmom, metaLRfo, betaLR, sinthetamom, sinthetafo = collect_data_n(d, r, T, train_n_list, test_n, reps)


# In[ ]:


save_data = {"metaLRmom" : metaLRmom, "metaLRfo" :  metaLRfo, "betaLR" : betaLR, "sinthetamom" : sinthetamom, "sinthetafo" : sinthetafo}


# In[ ]:


save_data["train_n_list"] =  train_n_list
save_data["d"] = d
save_data["r"] = r
save_data["T"] = T
save_data["test_n"] = test_n
save_data["reps"] = reps


# In[ ]:


params = "d="+str(d)+",r="+str(r)+",T="+str(T)+",test_n="+str(test_n)


# In[ ]:


file_name = "Meta,"+str(params)+".pickle"
folder_name = "Data"
file_path = os.path.join(folder_name, file_name)
pickle.dump(save_data, open(file_path, "wb"))


# In[ ]:




