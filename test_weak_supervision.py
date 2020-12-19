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
        
        #print(torch.min(explaination))
            
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
            #plt.show()
                    
        #ground_truth = ground_truth/torch.max(ground_truth)
        #explaination = explaination/torch.max(explaination)
        
        ground_truth = ground_truth.view(1,-1)
        explaination = explaination.view(1,-1)
        
        #print(ground_truth[0].shape)
        
        #print(explaination[0].cpu().data.numpy().tolist()[:10])
        #print(ground_truth[0].cpu().data.numpy().tolist()[:10])
        

        #if losses[-1] > model.percentile:
        cs = cosine_similarity(ground_truth, explaination)
        cs_list1.append(cs[0][0])

        cs_list2.append(top_k_similarity(explaination, ground_truth, k=30))

        cs_list3.append(top_k_similarity(explaination, ground_truth, k=2500))

        ground_truth = list(map(int,ground_truth[0].cpu().data.numpy().tolist()))
        explaination = explaination[0].cpu().data.numpy().tolist()
        cs_list4.append(roc_auc_score(ground_truth, explaination))

        if True:
            print(losses[-1])
            print(cs_list1[-1],cs_list2[-1],cs_list3[-1],cs_list4[-1], "\n")
            plt.show()
    
        
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
        #print(i)
        bi = base_images[i]#/np.max(base_images[i])
        ai = anomaly_images[i]#/np.max(anomaly_images[i])
        ground_truth = torch.Tensor((bi-ai)**2)
        
        ai = torch.Tensor(ai)
        R = LRP_a1b0(model, ai)
        r = R[0][0]
        explaination = r[0]
        
        #explaination = integrated_gradient(model, ai, b=1)
        #explaination = torch.Tensor(explaination)
        
        explaination = explaination.clamp(min=0)
        
        #print(torch.min(explaination))
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
                    
        #ground_truth = ground_truth/torch.max(ground_truth)
        #explaination = explaination/torch.max(explaination)
        
        ground_truth = ground_truth.view(1,-1)
        explaination = explaination.view(1,-1)
        
        if cosine:
            #cs = cosine_similarity(ground_truth, explaination)
            #cs_list.append(cs[0][0])
            cs_list.append(roc_auc_score(explaination, ground_truth))
        else:
            cs_list.append(top_k_similarity(explaination, ground_truth, k=2500))
        
        #if i%100==0:
        #    print(i/10000,"%")


    cs_list=np.array(cs_list)
    #print(anomaly,"explaination_acc:",np.mean(cs_list), "\tstd:",np.std(cs_list))
    #print(anomaly, "CLEVER HANS:", model.test_auc - np.mean(cs_list) , "\n")
    
    mean = np.mean(cs_list)
    std = np.std(cs_list)
        
    return mean, std


def load_model(path):
    P = path[:-6].split("/")
    S = P[-1].split("_")
    print(path)
        
    if S[1] == "none":
         model = DeepAD("DSVDD", "mvtec_vgg", "mvtec", epochs=0, nc=int(S[3]), loss=S[2], noise=random, do_pretrain=None, do_print=False)
         model.load_model(path)
    else:
        #model = DeepAD("DSVDD", "mnist_LeNet", "mnist", epochs=0, nc=-1,  loss=S[3], noise=random, do_pretrain=None, do_print=False)
        try: 
            model = DeepAD("DSVDD", "mvtec_vgg", "mvtec", epochs=0, nc=int(S[4]), loss=S[3], noise=random, do_pretrain=None, do_print=False)
            model.load_model(path)
        except:
            model = DeepAD("DSVDD", "mvtec_vgg_old", "mvtec", epochs=0, nc=int(S[4]), loss=S[3], noise=random, do_pretrain=None, do_print=False)
            model.load_model(path)


    
        
    return model
    


def mvtec_timeline():
    plt.switch_backend("TKAgg")
    loss = "normal"
    num_runs = 1
    c = 13
    
    mypath1 = "./results/"
    mypath2 = "./results2/"
    
    
    weak_sup = [1,2,4,8,16,32,64,256,1024,4096]
    
    oe = ["imagenet", "noise"]
    classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    print(classes[c])
    
    img_size = 224
    test_transform = [transforms.Resize(img_size),transforms.CenterCrop(img_size),transforms.ToTensor()]
    test_transform = transforms.Compose(test_transform)
    anomaly_images = MVTec(root='./datasets/mvtec_anomaly_detection/' + classes[c] + '/test', transform=test_transform, blur_outliers=True, blur_std=1.0)
    ground_truth_images = MVTec_Masks(root='./datasets/mvtec_anomaly_detection/' + classes[c] + '/ground_truth',transform=test_transform)
    
    anomaly = classes[c]
    print(len(ground_truth_images), len(anomaly_images))
    normal_count = 0
    for i in range(0, len(anomaly_images)):        
        ai, t, _ , _ = anomaly_images.__getitem__(i)
        
        if t!=0:
            ground_truth, _, _, _ = ground_truth_images.__getitem__(i-normal_count)
        
        ground_truth = (ground_truth[0]*3).clamp(max=1)
        
        if t == 0:
            normal_count += 1
            continue
        
       
        print(i)
    
        fig, ax = plt.subplots(len(oe),len(weak_sup)+2)
        
        ai_show = ai.permute(1,2,0)
        ai_show = ai_show[5:-5, 5:-5]
        ax[0, 0].imshow(ai_show)
        ax[0, 0].set_title("{}\n{}\nanomaly".format(anomaly, "imagenet"))
        ax[0 ,0].axis("off")
        ax[1, 0].imshow(ai_show)
        ax[1, 0].set_title("{}\n{}\nanomaly".format(anomaly, "noise"))
        ax[1, 0].axis("off")
        
        
        ax[0, 1].imshow(ground_truth[5:-5, 5:-5])
        ax[0, 1].axis("off")
        ax[0, 1].set_title("{}\n{}\nground truth".format(anomaly, "imagenet"))
        ax[1, 1].imshow(ground_truth[5:-5, 5:-5])
        ax[1, 1].axis("off")
        ax[1, 1].set_title("{}\n{}\nground truth".format(anomaly, "noise"))
        
        for w in range(len(weak_sup)):
            explanation = []
            for run in range(num_runs):
                try:
                    model_path = mypath1 + "MVTEC_imagenet_{}_{}_{}_{}.model".format(weak_sup[w],loss,c, run)
                    model = load_model(model_path)
                except:
                    model_path = mypath2 + "MVTEC_imagenet_{}_{}_{}_{}.model".format(weak_sup[w],loss,c, run)
                    model = load_model(model_path)
                    
                R = LRP_a1b0(model, ai)
                r = R[0][0]
                explaination = r[0]
                explaination = explaination.clamp(min=0)
                explanation.append(explaination[5:-5, 5:-5])
                del model
                gc.collect()
            
            explanation = torch.stack(explanation)
            explanation = torch.mean(explanation, dim=0)
            ax[0, w+2].imshow(explanation)
            ax[0, w+2].axis("off")
            ax[0, w+2].set_title("{}".format(weak_sup[w]))
            
            
            explanation = []
            for run in range(num_runs):
                try:
                    model_path = mypath1 + "MVTEC_noise_{}_{}_{}_{}.model".format(weak_sup[w],loss,c, run)
                    model = load_model(model_path)
                except:
                    model_path = mypath2 + "MVTEC_noise_{}_{}_{}_{}.model".format(weak_sup[w],loss,c, run)
                    model = load_model(model_path)
                R = LRP_a1b0(model, ai)
                r = R[0][0]
                explaination = r[0]
                explaination = explaination.clamp(min=0)
                explanation.append(explaination)
                del model
                gc.collect()
            
            explanation = torch.stack(explanation)
            explanation = torch.mean(explanation, dim=0)
            ax[1, w+2].imshow(explanation)
            ax[1, w+2].axis("off")
            ax[1, w+2].set_title("{}".format(weak_sup[w]))

        plt.show()




def main():
    csv = {}
    
    with open("results_mnistc_bce.json", 'r+') as json_file:
        csv = json.load(json_file)
    
    out = int(sys.argv[1])
    out_file_create = open("./jsonresults_bce/results"+str(out)+".json","x")
    
    folders = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]
    
    mypath = "./results/"
    #mypath="/home/julius.coburger/exp/master_thesis/results/"
    paths = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    random.shuffle(paths)
    
    #paths = ["MNISTC_emnist_16384_bce_0.model","MNISTC_mnistc_32_bce_4.model"]
        
    for path in paths:
        print(path)
        
        split = path.split("/")
        p = split[-1]

            
        model_csv = []
        
        if (p not in csv) and ("results" not in p) and ("MNISTC" in p) and ("bce" in p):
        
            #load model
            model = load_model("./results/" + path)
    
            
            for f in folders:
                print(f)
                test_point = [f]
                
                #compute and save auc on given anomaly  
                model.dataset.load_new_test_set([f])
                model.test()
                test_point.append(str(model.test_auc))
                                
                #compute and save cosine cosine_similarity
                mean, std = test_cosine_mnist(model,anomaly=f, cosine=True)
                test_point.append(str(mean))
                test_point.append(str(std))
                
                #compute and save top-30-cosine_similarity
                mean, std = test_cosine_mnist(model,anomaly=f, cosine=False)
                test_point.append(str(mean))
                test_point.append(str(std))

                model_csv.append(test_point)
                                
            
            csv[p] = model_csv


            json_out = json.dumps(csv)
            f = open("./jsonresults_bce/results"+str(out)+".json","w")
            f.write(json_out)
            f.close()
            
            del model
            gc.collect()


def main_mvtec():
    csv = {}
    
    #with open("results_mvtec3.json", 'r+') as json_file:
    #    csv = json.load(json_file)
    
    #out = int(sys.argv[1]) + 120
    #out_file_create = open("./jsonresults_mvtec/results"+str(out)+".json","x")
    
   
    mypath = "./results/"
    #mypath="/home/julius.coburger/exp/master_thesis/results/"
    paths = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    random.shuffle(paths)
    
    #paths = ["MVTEC_imagenet_8_bce_6_1.model", "MVTEC_imagenet_8_bce_2_3.model", "MVTEC_imagenet_32_bce_13_0.model"] 
    #paths = ["MVTEC_imagenet_16_hypersphere_4_0.model", "MVTEC_noise_256_hypersphere_6_0.model", "MVTEC_imagenet_8_bce_6_1.model" , "MVTEC_imagenet_32_bce_13_0.model"]
    
    paths = ["MVTEC_confetti_4096_hypersphere_13_5.model"]
        
    for path in paths:
        print(path)
        
        split = path.split("/")
        p = split[-1]

            
        #model_csv = []
        
        if (p not in csv) and ("results" not in p) and ("MVTEC" in p):
        
            #load model
            model = load_model("./results/" + path)
            test_point = [model.nc]
            
            #compute and save auc on given anomaly  
            model.test()
            print(model.percentile)
            test_point.append(str(model.test_auc))

            #compute and save cosine cosine_similarity
            r = test_cosine_mvtec(model,normal_class=model.nc)
            test_point.extend(list(map(str,r)))
            print(test_point)
            
            
            csv[p] = test_point

            #print(csv)

            #json_out = json.dumps(csv)
            #f = open("./jsonresults_mvtec/results"+str(out)+".json","w")
            #f.write(json_out)
            #f.close()
            
            del model
            gc.collect()




if __name__ == '__main__':
    
    #main()
    main_mvtec()
    #mvtec_timeline()

    

