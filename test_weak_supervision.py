import json
import random
import torch
import time
import gc
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns               #unused?
from datasets.main import load_dataset
from DeepAD import DeepAD
from PIL import Image, ImageDraw    #unused?
from explain import grads, integrated_gradient, LRP, top_k_similarity
from datasets.MVTec import MVTec, MVTec_Masks
import torchvision.transforms as transforms
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity



def test_cosine_mvtec(model, normal_class=0):
    classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    print(classes[normal_class])
    
    img_size = 224
    test_transform = [transforms.Resize(img_size),transforms.CenterCrop(img_size),transforms.ToTensor()]
    
    test_transform = transforms.Compose(test_transform)
    
    anomaly_images = MVTec(root='./datasets/mvtec_anomaly_detection/' + classes[normal_class] + '/test', transform=test_transform, blur_outliers=True, blur_std=1.0)
    
    ground_truth_images = MVTec_Masks(root='./datasets/mvtec_anomaly_detection/' + classes[normal_class] + '/ground_truth',transform=test_transform)
    
    cs_list1 = []
    cs_list2 = []
    cs_list3 = []
    cs_list4 = []
    
    losses = []
    
    print(len(ground_truth_images), len(anomaly_images))
    normal_count = 0
    for i in range(0, len(anomaly_images)):
        
        ai, t, _ , _ = anomaly_images.__getitem__(i)
        
        if t!=0:
            ground_truth, _, _, _ = ground_truth_images.__getitem__(i-normal_count)
        
        if t == 0:
            normal_count += 1
            continue
        
        ground_truth = (ground_truth[0]*3).clamp(max=1)
        R = LRP_a1b0(model, ai)
        r = R[0][0]
        losses.append(torch.sum(R[-1][0]))
        explaination = r[0]
        
        #explaination = integrated_gradient(model, ai, b=1)
        #explaination = torch.Tensor(explaination)
        
        explaination = explaination.clamp(min=0)
        
            
        if False:
            plt.switch_backend("TKAgg")
            #print(t, i)
            fig, ax = plt.subplots(1,3)
            cm="gray_r"
            ai = ai.permute(1,2,0)
            ax[0].set_title("anomaly")
            ax[1].set_title("ground_truth")
            ax[2].set_title("explaination")
            ax[0].imshow(ai)
            ax[1].imshow(ground_truth)
            ax[2].imshow(explaination)
            plt.show()

                    

        
        ground_truth = ground_truth.view(1,-1)
        explaination = explaination.view(1,-1)

        cs = cosine_similarity(ground_truth, explaination)
        cs_list1.append(cs[0][0])

        cs_list2.append(top_k_similarity(explaination, ground_truth, k=30))

        cs_list3.append(top_k_similarity(explaination, ground_truth, k=2500))

        ground_truth = list(map(int,ground_truth[0].cpu().data.numpy().tolist()))
        explaination = explaination[0].cpu().data.numpy().tolist()
        cs_list4.append(roc_auc_score(ground_truth, explaination))

        if False:
            print(losses[-1])
            print(cs_list1[-1],cs_list2[-1],cs_list3[-1],cs_list4[-1], "\n")
    
        
    losses = np.array(losses)

    cs_list1=np.array(cs_list1)
    mean1 = np.mean(cs_list1)
    std1 = np.std(cs_list1)
    
    cs_list2=np.array(cs_list2)
    mean2 = np.mean(cs_list2)
    std2 = np.std(cs_list2)
    
    cs_list3=np.array(cs_list3)
    mean3 = np.mean(cs_list3)
    std3 = np.std(cs_list3)
    
    cs_list4=np.array(cs_list4)
    mean4 = np.mean(cs_list4)
    std4 = np.std(cs_list4)

    return [mean1, std1, mean2, std2, mean3, std3, mean4, std4]





def test_cosine_mnist(model, anomaly="zigzag", cosine=True):
    base_images = np.load("./datasets/MNIST-C/mnist_c/identity/test_images.npy")
    base_images = np.squeeze(base_images)/255
    anomaly_images = np.load("./datasets/MNIST-C/mnist_c/"+anomaly+"/test_images.npy")
    anomaly_images = np.squeeze(anomaly_images)/255
    
    cs_list = []
    
    for i in range(10000):
        bi = base_images[i]#/np.max(base_images[i])
        ai = anomaly_images[i]#/np.max(anomaly_images[i])
        ground_truth = torch.Tensor((bi-ai)**2)
        
        ai = torch.Tensor(ai)
        R = LRP_a1b0(model, ai)
        r = R[0][0]
        explaination = r[0]
   
        
        explaination = explaination.clamp(min=0)
        
   
        if False:
            fig, ax = plt.subplots(1,4)
            cm="gray_r"
            ax[0].set_title("base_image")
            ax[1].set_title("anomaly")
            ax[2].set_title("ground_truth")
            ax[3].set_title("explaination")
            ax[0].imshow(base_images[i],vmin=0, vmax=1, cmap=cm)
            ax[1].imshow(anomaly_images[i],vmin=0, vmax=1, cmap=cm)
            ax[2].imshow(ground_truth, cmap=cm)
            ax[3].imshow(explaination, cmap=cm)
            plt.show()
                
        
        ground_truth = ground_truth.view(1,-1)
        explaination = explaination.view(1,-1)
        
        if cosine:
            cs_list.append(roc_auc_score(explaination, ground_truth))
        else:
            cs_list.append(top_k_similarity(explaination, ground_truth))


    cs_list=np.array(cs_list)
    
    mean = np.mean(cs_list)
    std = np.std(cs_list)
        
    return mean, std



