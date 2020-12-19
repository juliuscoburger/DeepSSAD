from datasets.main import load_dataset
from DeepAD import DeepAD
import numpy as np
import random
import torch
import time
import seaborn as sns
from explain import grads, pixel_flipping, integrated_gradient, LRP_a1b0, LRP, plot_explaination, top_k_similarity
import pandas as pd
from datasets.MVTec import MVTec, MVTec_Masks
import torchvision.transforms as transforms
import sys
import os


from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
import gc
from os import listdir
from os.path import isfile, join






def pipe(c=2, net_name="mnist_LeNet", dataset_name="mnist", loss="normal", do_print=True, do_pretrain=True, noise="random", weak_supervision=True, weak_supervision_size=2,  weak_supervision_set="mnistc", out_file=None):

    #setup net
    model = DeepAD("DSVDD", net_name, d_name=dataset_name, epochs=100, lr=1e-4, nc=c, lr_milestones=[70], loss=loss, do_print=do_print, do_pretrain=do_pretrain, pretrain_epochs=20, noise=noise, weak_supervision=weak_supervision, weak_supervision_size=weak_supervision_size, device="cpu", weak_supervision_set=weak_supervision_set, out_file=out_file)

    
    #train
    model.train()
    #print("Train duration:", model.train_time)
    

    #test
    model.test()
    if do_print:
        print("AUC_:",model.test_auc)
    
    
    return model


    
   

    
def load_model(path):
    P = path[:-6].split("/")
    S = P[-1].split("_")
    
    noise="none"
    if len(S) >4:
        noise = S[4]
    
        if len(S) > 5:
            noise = S[4]+S[5]
        
    d=S[1].lower()
    
    if d == "mvtec":
        model = DeepAD(S[0], "mvtec_vgg", d, epochs=0, nc=int(S[2]), loss=S[3], noise=noise, do_pretrain=None, do_print=False)
    else:
        model = DeepAD(S[0], d+"_LeNet", d, epochs=0, nc=int(S[2]),  loss=S[3], noise=noise, do_pretrain=None, do_print=False)
    model.load_model(path)
        
    return model
        
    
    
    

def train_mnistc():
    #os.environ['CUDA_VISIBLE_DEVICES']='2'

    losses = ["bce"]
    anom = ["emnist", "mnistc"]
    num_runs = 5
    weak_supervision = int(sys.argv[1])
    
    
    for loss in losses:
        for a in anom:
            for i in range(num_runs):
                model = pipe(c=-1, net_name="mnist_LeNet", dataset_name="mnist", loss=loss, do_print=True, do_pretrain=True, 
                             noise="random", 
                             weak_supervision=True,
                             weak_supervision_size=weak_supervision, 
                             weak_supervision_set=a)
                
                #model_name = {anomaly-type}_{anomaly-amount}_{loss}_{run}
                model.save_model("results/MNISTC_{}_{}_{}_{}.model".format(a, model.weak_supervision_size, loss, i))
                del model
                gc.collect()

    return



def train_mvtec():
    #os.environ['CUDA_VISIBLE_DEVICES']='2'
    
    mypath = "./results/"
    #mypath="/home/julius.coburger/exp/master_thesis/results/"
    paths = [f for f in listdir(mypath) if isfile(join(mypath, f))]


    losses = ["bce"]
    num_runs = 5
    weak_supervision = int(sys.argv[1])
    classes = 15
    
    #noise weak supervision
    for c in range(classes):
        for loss in losses:
            for i in range(num_runs):
                a = "noise"
                name = "MVTEC_{}_{}_{}_{}_{}.model".format(a, weak_supervision, loss, c, i)
                print(name)
                if name not in paths:
                    model = pipe(c=c, net_name="mvtec_vgg", dataset_name="mvtec", loss=loss, do_print=False, do_pretrain=False, 
                                    noise="random", 
                                    weak_supervision=True,
                                    weak_supervision_size=weak_supervision, 
                                    weak_supervision_set="mvtec_noise")
                    
                    #model_name = {anomaly-type}_{anomaly-amount}_{loss}_{run}
                    a = "noise"
                    model.save_model("results/MVTEC_{}_{}_{}_{}_{}.model".format(a, model.weak_supervision_size, loss, c, i))
                    del model
                    gc.collect()

            
    #no supervision
    for c in range(classes):
        for loss in losses:
            for i in range(num_runs):
                a = "none"
                name = "MVTEC_{}_{}_{}_{}_{}.model".format(a, weak_supervision, loss, c, i)
                if name not in paths:
                    model = pipe(c=c, net_name="mvtec_vgg", dataset_name="mvtec", loss=loss, do_print=False, do_pretrain=False, 
                                    noise="random", 
                                    weak_supervision=False)
                    
                    #model_name = {anomaly-type}_{anomaly-amount}_{loss}_{run}
                    a = "none"
                    model.save_model("results/MVTEC_{}_{}_{}_{}_{}.model".format(a, model.weak_supervision_size, loss, c, i))
                    del model
                    gc.collect()

                


    return
    
    
    

    
def train_mvtec_imagenet():    
    mypath = "./results/"
    #mypath="/home/julius.coburger/exp/master_thesis/results/"
    paths = [f for f in listdir(mypath) if isfile(join(mypath, f))]


    num_runs = 5
    weak_supervision = int(sys.argv[1])
    loss= str(sys.argv[2])
    c = int(sys.argv[3])
    
    #noise weak supervision
    
    for i in range(num_runs):
        a = "imagenet"
        name = "MVTEC_{}_{}_{}_{}_{}.model".format(a, weak_supervision, loss, c, i)
        print(name)
        if name not in paths:
            model = pipe(c=c, net_name="mvtec_vgg", dataset_name="mvtec", loss=loss, do_print=True, do_pretrain=False, 
                            noise="random", 
                            weak_supervision=True)
            
            #model_name = {anomaly-type}_{anomaly-amount}_{loss}_{run}
            a = "imagenet"
            model.save_model("results2/MVTEC_{}_{}_{}_{}_{}.model".format(a, model.weak_supervision_size, loss, c, i))
            del model
            gc.collect()
            
            
            
def train_mvtec_noise():    
    mypath = "./results/"
    #mypath="/home/julius.coburger/exp/master_thesis/results/"
    paths = [f for f in listdir(mypath) if isfile(join(mypath, f))]


    num_runs = 5
    weak_supervision = int(sys.argv[1])
    loss= str(sys.argv[2])
    c = int(sys.argv[3])
    
    #noise weak supervision
    
    for i in range(num_runs):
        a = "imagenet"
        name = "MVTEC_{}_{}_{}_{}_{}.model".format(a, weak_supervision, loss, c, i)
        print(name)
        if name not in paths:
            model = pipe(c=c, net_name="mvtec_vgg", dataset_name="mvtec", loss=loss, do_print=True, do_pretrain=False, 
                            noise="random", 
                            weak_supervision=True,
                            weak_supervision_size=weak_supervision, 
                            weak_supervision_set="mvtec_imagenet")
            
            #model_name = {anomaly-type}_{anomaly-amount}_{loss}_{run}
            a = "imagenet"
            model.save_model("results2/MVTEC_{}_{}_{}_{}_{}.model".format(a, model.weak_supervision_size, loss, c, i))
            del model
            gc.collect()
            
            
    if weak_supervision==4096:
        for i in range(num_runs):
            a = "none"
            name = "MVTEC_{}_{}_{}_{}_{}.model".format(a, weak_supervision, loss, c, i)
            print(name)
            if name not in paths:
                model = pipe(c=c, net_name="mvtec_vgg", dataset_name="mvtec", loss=loss, do_print=True, do_pretrain=False, 
                                noise="random", 
                                weak_supervision=False,
                                weak_supervision_size=weak_supervision, 
                                weak_supervision_set="mvtec_noise")
                
                #model_name = {anomaly-type}_{anomaly-amount}_{loss}_{run}
                a = "none"
                model.save_model("results2/MVTEC_{}_{}_{}_{}.model".format(a, loss, c, i))
                del model
                gc.collect()
    
    

def main():
    #train_mnistc()
    #train_mvtec_imagenet()
    #train_mvtec_noise()
    loss = "hypersphere"
    weak_supervision = 8
    c = 3
    a = "confetti"
    i = 5
    name = "test.res"
    f = open(name, "a")
    sys.stdout = f
    print(name, file=f)
    
    model = pipe(c=c, net_name="mvtec_vgg_unfreeze", dataset_name="mvtec", loss=loss, do_print=True, do_pretrain=False, out_file=f,
                            noise="random", 
                            weak_supervision=True,
                            weak_supervision_size=weak_supervision, 
                            weak_supervision_set="mvtec_noise")
            
    #model_name = {anomaly-type}_{anomaly-amount}_{loss}_{run}
    a = "confetti"
    model.save_model("results/MVTEC_{}_{}_{}_{}_{}.model".format(a, model.weak_supervision_size, loss, c, i))
            

if __name__ == '__main__':
    main()
    
    
