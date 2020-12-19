from datasets.main import load_dataset
from DeepAD import DeepAD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
import random
import torch
import time
import seaborn as sns
from explain import grads, pixel_flipping, integrated_gradient, LRP_a1b0, LRP, plot_explaination, top_k_similarity
import pandas as pd
from datasets.MVTec import MVTec, MVTec_Masks
import torchvision.transforms as transforms



from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
import gc

plt.switch_backend("TKAgg")




'''
TODOs
    -load existing model
    
    -preprocessing
    
    -cosmetics 
    -save results

    -using different devices
'''



def pipe(c=2, net_name="mnist_LeNet", dataset_name="mnist", loss="normal", do_print=True, do_pretrain=True, noise="random", epochs=100):

    #setup net
    model = DeepAD("DSVDD", net_name, d_name=dataset_name, epochs=epochs, lr=1e-4, nc=c, lr_milestones=[70], loss=loss, do_print=do_print, do_pretrain=do_pretrain, pretrain_epochs=20, noise=noise, weak_supervision=True, weak_supervision_size=32, weak_supervision_set="mvtec_noise")

    
    #train
    model.train()
    #print("Train duration:", model.train_time)
    

    #test
    model.test()
    if do_print:
        print("AUC_:",model.test_auc)
    
    #model.save_model("results/DWSAD-{}_MNIST_{}_normal_random.model".format(model.weak_supervision_size, c))

    
    return model

    
    #----------------------------------------------------------------------------------------------------------
    
    
    
    
    #individual test
    images = []
    labels = []
    outputs = []
    
    while len(images) < 10:
        img, l, _, _ = model.dataset.test_set.__getitem__(random.randint(0, 1000))
        if (len(images) < 5 and l == 0) or len(images)>=5 and l==1:
            out = model.net(img)
            dist = torch.sum((out - model.c) ** 2, dim=1)
            loss = torch.mean(dist)
            outputs.append(loss)
            images.append(img)
            labels.append(l)
            
            
    print(images[0].shape)
    #return model 
    
    fig, ax = plt.subplots(2,5)
    ax = ax.flatten()
    
        
    for i in range(10):
        ax[i].set_title('Label: {}\n Dist: {:.2f}'.format(labels[i], outputs[i]))
        
        if model.dataset_name == "cifar10":
            ax[i].imshow(np.transpose(images[i],(1, 2, 0)), cmap='gray_r')
        else:
            ax[i].imshow(images[i].reshape((28,28)), cmap='gray_r')
            
    plt.tight_layout()
    plt.show()
    
    
    

    fig, ax = plt.subplots(3,8)
    colormap1 = "seismic_r"
    samples = [images[0], images[5], images[6]]
    titles = ["Image", "Gradient", "Gradient*Input", "Senitivity A", "LRP", "LRP_a1b0", "LRP_deletion", "IntGradient"]


    if model.dataset_name != "cifar10":
        samples = [s.reshape((28,28)) for s in samples]
    
    for i in range(len(samples)):
        img, g1,g2,g3 = grads(model, samples[i])

        
        if model.dataset_name == "cifar10":
            img = np.transpose(img,(1,2,0))
            g1 = torch.mean(g1, dim=0)
            g2 = torch.mean(g2, dim=0)
            g3 = torch.mean(g3, dim=0)

            
        
        #print(samples[i])
        
        R = LRP(model, samples[i])
        r = R[0].cpu().detach().numpy()[0]
        
        R2 = LRP_a1b0(model, samples[i])
        r2 = R2[0].cpu().detach().numpy()[0]
        
        
        iG = integrated_gradient(model, samples[i])
        
        
        b = [10*((np.abs(a)**3.0).mean()**(1.0/3)) for a in [g1,g2,g3,r[0],r2[0], iG]]
        
        print(iG.shape, "IG shape")
        
        
        
        for j in range(1,8):
            ax[i, j].set_title('{}'.format(titles[j]))
            
        
        out = model.net(torch.stack(samples))
        dist = torch.sum((out - model.c) ** 2, dim=1)
        ax[i,0].set_title("{}\n{:.2f}".format(titles[0], dist[i]))
            
        ax[i,0].imshow(img, cmap="gray_r")
        ax[i,1].imshow(g1 , cmap=colormap1,vmin=-b[0],vmax=b[0],interpolation='nearest')
        ax[i,2].imshow(g2,  cmap=colormap1,vmin=-b[1],vmax=b[1],interpolation='nearest')
        ax[i,3].imshow(g3,  cmap=colormap1,vmin=-b[2],vmax=b[2],interpolation='nearest')
        ax[i,4].imshow(r[0],cmap=colormap1,vmin=-b[3],vmax=b[3],interpolation='nearest')
        ax[i,5].imshow(r2[0],cmap=colormap1,vmin=-b[3],vmax=b[3],interpolation='nearest')
        ax[i,6].imshow(pixel_flipping(img, r[0], amount=30) ,cmap="gray_r")
        ax[i,7].imshow(iG,cmap=colormap1,vmin=-b[4],vmax=b[4],interpolation='nearest')

    
    #plt.tight_layout()
    plt.show()
    
    
    
    #test self created mnist anomalies
    #test_anomalies(model, model.nc)

     
    #save results   #save model
    s_time = int(time.time())
    model.save_results("results/results_"+ str(s_time) +".txt")
    model.save_model("results/model_"+ str(s_time) +".model")
    
    return model


#only test for images of normal class with the added anomaly
def test_anomalies(model, normal_class, amount=70):
    colormap1 = "seismic_r"
    titles = ["Image", "Gradient", "Gradient*Input", "Senitivity A", "LRP"]
    
    samples = []
    for i in range(1000):
        img = Image.open("./anomaly_pictures/{}_{}.png".format(i, normal_class))
        samples.append(torch.tensor(np.array(img)).float())
        
    fig, ax = plt.subplots(4,7)
    
    for i in range(4):
        print(i)
        img, g1,g2,g3 = grads(model, samples[i])
        
        if model.dataset_name == "cifar10":
            img = np.transpose(img,(1,2,0))
            g1 = torch.mean(g1, dim=0)
            g2 = torch.mean(g2, dim=0)
            g3 = torch.mean(g3, dim=0)

            
        
        R = LRP(model, samples[i])
        r = R[0].cpu().detach().numpy()[0]
        iG = integrated_gradient(model, samples[i])
        b = [10*((np.abs(a)**3.0).mean()**(1.0/3)) for a in [g1,g2,g3,r[0], iG]]
        
        
        for j in range(1,5):
            ax[i, j].set_title('{}'.format(titles[j]))
            
        
        out = model.net(torch.stack(samples))
        dist = torch.sum((out - model.c) ** 2, dim=1)
        ax[i,0].set_title("{}\n{:.2f}".format(titles[0], dist[i]))
        

        ax[i,0].imshow(img, cmap="gray_r")
        ax[i,1].imshow(g1 , cmap=colormap1,vmin=-b[0],vmax=b[0],interpolation='nearest')
        ax[i,2].imshow(g2,  cmap=colormap1,vmin=-b[1],vmax=b[1],interpolation='nearest')
        ax[i,3].imshow(g3,  cmap=colormap1,vmin=-b[2],vmax=b[2],interpolation='nearest')
        ax[i,4].imshow(r[0],cmap=colormap1,vmin=-b[3],vmax=b[3],interpolation='nearest')
        ax[i,5].imshow(pixel_flipping(img, r[0], amount=70) ,cmap="gray_r")
        ax[i,6].imshow(iG,  cmap=colormap1,vmin=-b[4],vmax=b[4],interpolation='nearest')
        
        px_out = model.net(torch.tensor(pixel_flipping(img, r[0], amount=70)))
        px_dist = torch.sum((px_out - model.c) ** 2, dim=1)
        
        #print("{:.2f}  -> {:.2f}".format(np.sqrt(dist[i].detach().numpy()), np.sqrt(px_dist.detach().numpy()[0])))
        
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
    out = model.net(torch.stack(samples))
    sum_dist = torch.sum(torch.sum((out - model.c) ** 2, dim=1))
    
    lrp_samples = []
    g1_samples = []
    g2_samples = []
    g3_samples = []
    
    for i in range(len(samples)):
        img, g1,g2,g3 = grads(model, samples[i])
        
        R = LRP(model, samples[i])
        r = R[0].cpu().detach().numpy()[0]
        
        
        g1_samples.append(torch.tensor(pixel_flipping(img, g1, amount=amount)))
        g2_samples.append(torch.tensor(pixel_flipping(img, g2, amount=amount)))
        g3_samples.append(torch.tensor(pixel_flipping(img, g3, amount=amount)))
        lrp_samples.append(torch.tensor(pixel_flipping(img, r[0], amount=amount)))
        

      
    lrp_out = model.net(torch.stack(lrp_samples))
    lrp_sum_dist = torch.sum(torch.sum((lrp_out - model.c) ** 2, dim=1))
    
    g1_out = model.net(torch.stack(g1_samples))
    g1_sum_dist = torch.sum(torch.sum((g1_out - model.c) ** 2, dim=1))
    
    g2_out = model.net(torch.stack(g2_samples))
    g2_sum_dist = torch.sum(torch.sum((g2_out - model.c) ** 2, dim=1))
    
    g3_out = model.net(torch.stack(g3_samples))
    g3_sum_dist = torch.sum(torch.sum((g3_out - model.c) ** 2, dim=1))
    
    l = [sum_dist.detach().numpy(), lrp_sum_dist.detach().numpy(), g1_sum_dist.detach().numpy(), g2_sum_dist.detach().numpy(), g3_sum_dist.detach().numpy()]
        
    
    print("Dist:\t{}\n LRP:\t{}\n G1:\t{}\n G2:\t{}\n G3:\t{}\n".format(np.sqrt(l[0]), np.sqrt(l[1]), np.sqrt(l[2]), np.sqrt(l[3]), np.sqrt(l[4])))
        
    
    return
    
    
    
def plot_densities(model):
    _, labels, scores = zip(*model.test_scores)
    
    scores_normal = [scores[i] for i in range(len(scores)) if labels[i]==0]
    scores_anomaly = [scores[i] for i in range(len(scores)) if labels[i]==1]
    
    fig, ax = plt.subplots(2)
    ax[0].set_title("Normal samples score distribution")
    ax[0].hist(scores_normal, bins=100)
    
    ax[1].set_title("Anomaly samples score distribution")
    ax[1].hist(scores_anomaly, bins=100)
    plt.tight_layout()
    plt.show()
    
    return



def get_most_normal_anomalies(model):
    #only get from specfic class
    idx, labels, scores = zip(*model.test_scores)
    
    idx2 = [idx[i] for i in range(len(scores)) if labels[i]==0]
    scores2 = [scores[i] for i in range(len(scores)) if labels[i]==0]
    
    
    ind = np.argsort(scores2)
    
    maxTest = ind[-5:]
    minTest = ind[:5]
    
    print(maxTest)
    print(minTest)
    
    fig, ax = plt.subplots(2,5)
    for i in range(5):
        ax[0,i].set_title("{:.2f}".format(scores2[minTest[i]]))
        
        img,_,_,_ = model.dataset.test_set.__getitem__(idx2[minTest[i]])
        ax[0,i].imshow(img[0], cmap="gray_r")
        
        
        ax[1,i].set_title("{:.2f}".format(scores2[maxTest[i]]))        
        img,_,_,_ = model.dataset.test_set.__getitem__(idx2[maxTest[i]])
        ax[1,i].imshow(img[0], cmap="gray_r")
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    for i in range(5):
    
        fig, ax = plt.subplots(2,6)
        
        #analyzis on the most normal picture
        img,_,_,_ = model.dataset.test_set.__getitem__(idx2[minTest[0]])
        img = img[0]
        
        ax[0,0].set_title("Original\n Score={:.3f}".format(scores2[minTest[0]]))
        ax[0,0].imshow(img, cmap="gray_r")
        
        if i ==0:
            img[2,2] = 1
            img[3,3] = 1
            img[3,2] = 1
            img[2,3] = 1
        else:
            coords = [random.randint(0,27) for j in range(4)]
            coords.sort()
            print(coords)
            if coords[1]==0:
                coords[1] = 10
            a = (coords[3]-coords[2])/(coords[1]-coords[0])
            for x in range(coords[1]-coords[0]):
                y = a*x + coords[2]
                img[x+coords[0], int(y)] = 1
            
        
        out = model.net(img)
        dist = torch.sum((out - model.c) ** 2, dim=1)
        
        
        ax[0,1].set_title("Modified\n Score={:.3f}".format(dist[0]))
        ax[0,1].imshow(img, cmap="gray_r")
        
        iG = integrated_gradient(model, img)
        R = LRP(model, img)
        r = R[0].cpu().detach().numpy()[0]
        
        R2 = LRP_a1b0(model, img)
        r2 = R2[0].cpu().detach().numpy()[0]
        
        img3, g1,g2,g3 = grads(model, img)

        
        
        #cm = "bone_r"
        cm = "seismic_r"
        
        ax[0,2].set_title("Grad")
        b = [torch.max(g1), -1*torch.min(g1)]
        ax[0,2].imshow(g1, cmap=cm, vmin=-1*max(b), vmax=max(b))
        
        ax[0,3].set_title("Grad*Input")
        b = [torch.max(g2), -1*torch.min(g2)]
        ax[0,3].imshow(g2, cmap=cm, vmin=-1*max(b), vmax=max(b))
        
        ax[1,1].set_title("Salency Map")
        b = [torch.max(g3), -1*torch.min(g3)]
        ax[1,1].imshow(g3, cmap=cm, vmin=-1*max(b), vmax=max(b))
        
        ax[1,2].set_title("iG")
        b = [np.max(iG), -1*np.min(iG)]
        ax[1,2].imshow(iG, cmap=cm, vmin=-1*max(b), vmax=max(b))
        
        ax[1,3].set_title("LRP")
        b = [np.max(r[0]), -1*np.min(r[0])]
        ax[1,3].imshow(r[0], cmap="seismic_r", vmin=-1*max(b), vmax=max(b))
        
        
        
        
        ax[0,5].set_title("LRP_a1b0")
        b = [np.max(r2[0]), -1*np.min(r2[0])]
        ax[0,5].imshow(r2[0], cmap="seismic_r", vmin=-1*max(b), vmax=max(b))
        
 
            
        ax[1,0].axis("off")
        ax[1,5].axis("off")
        
        
        iG = integrated_gradient(model, img, b=1)
        ax[0,4].set_title("iG_2")
        b = [np.max(iG), -1*np.min(iG)]
        ax[0,4].imshow(np.abs(iG), cmap=cm, vmin=-1*max(b), vmax=max(b))
        
        
        iG = integrated_gradient(model, img, b=2)
        ax[1,4].set_title("iG_3")
        b = [np.max(iG), -1*np.min(iG)]
        ax[1,4].imshow(np.abs(iG), cmap=cm, vmin=-1*max(b), vmax=max(b))
        
        
        plt.tight_layout()
        plt.show()
        
        plot_explaination(model, img)
    
    return



def test_cosine_mnist(model, anomaly="zigzag", cosine=True):
    base_images = np.load("./datasets/MNIST-C/mnist_c/identity/test_images.npy")
    base_images = np.squeeze(base_images)/255
    anomaly_images = np.load("./datasets/MNIST-C/mnist_c/"+anomaly+"/test_images.npy")
    anomaly_images = np.squeeze(anomaly_images)/255
    
    cs_list = []
    
    for i in range(10000):
        bi = base_images[i]#/np.max(base_images[i])
        ai = anomaly_images[i]#/np.max(anomaly_images[i])
        ground_truth = (bi-ai)**2
        
        
        fig, ax = plt.subplots(1,2)
        ax[0].set_title("anomaly")
        ax[1].set_title("ground_truth")
        ax[0].imshow(ai)
        ax[1].imshow(ground_truth)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        ax[1].axes.xaxis.set_visible(False)
        ax[1].axes.yaxis.set_visible(False)
        plt.show()  
        continue
        
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
            cs = cosine_similarity(ground_truth, explaination)
            cs_list.append(cs[0][0])
        else:
            cs_list.append(top_k_similarity(explaination, ground_truth))
        
        #if i%100==0:
        #    print(i/10000,"%")


    cs_list=np.array(cs_list)
    print(anomaly,"explaination_acc:",np.mean(cs_list), "\tstd:",np.std(cs_list))
    print(anomaly, "CLEVER HANS:", model.test_auc - np.mean(cs_list) , "\n")
        
    return



def test_cosine_mvtec(model, normal_class=0, cosine=True):
    classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                   'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    img_size = 224
    test_transform = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()]
    
    test_transform = transforms.Compose(test_transform)
    
    anomaly_images = MVTec(root='./datasets/mvtec_anomaly_detection/' + classes[normal_class] + '/test',
                              transform=test_transform, blur_outliers=True, blur_std=1.0)
    
    ground_truth_images = MVTec_Masks(root='./datasets/mvtec_anomaly_detection/' + classes[normal_class] + '/ground_truth',
                              transform=test_transform)
    
    cs_list = []
    
    for i in range(0, len(ground_truth_images)):
        
        ai, t, _ , _ = anomaly_images.__getitem__(i)
        ground_truth, _, _, _ = ground_truth_images.__getitem__(i)
        
        if t == 0:
            continue
        
        ground_truth = ground_truth[0]
        fig, ax = plt.subplots(1,2)
        ax[0].set_title("anomaly")
        ax[1].set_title("ground_truth")
        ai = ai.permute(1,2,0)
        ax[0].imshow(ai)
        ax[1].imshow(ground_truth)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        ax[1].axes.xaxis.set_visible(False)
        ax[1].axes.yaxis.set_visible(False)
        plt.show()  
        continue
        
        #ai = ai.permute(1,2,0)
        ground_truth = ground_truth[0]

        R = LRP_a1b0(model, ai)
        r = R[0][0]
        explaination = r.sum(axis=0)
        
        explaination = explaination.clamp(min=0)
        
        
        if False:
            fig, ax = plt.subplots(1,3)
            ax[0].set_title("anomaly")
            ax[1].set_title("ground_truth")
            ax[2].set_title("explaination")
            ai = ai.permute(1,2,0)
            ax[0].imshow(ai)
            ax[1].imshow(ground_truth)
            ax[2].imshow(explaination)
            plt.show()    
            
            
        ground_truth = ground_truth.view(1,-1)
        explaination = explaination.view(1,-1)
        #cs = cosine_similarity(ground_truth, explaination)
        #cs_list.append(cs[0][0])
        
        if cosine:
            cs = cosine_similarity(ground_truth, explaination)
            cs_list.append(cs[0][0])
        else:
            cs_list.append(top_k_similarity(explaination, ground_truth))
        

    cs_list=np.array(cs_list)
    anomaly = classes[normal_class]
    print(anomaly,"explaination_acc:",np.mean(cs_list), "\tstd:",np.std(cs_list))
    print(anomaly, "CLEVER HANS:", model.test_auc - np.mean(cs_list) , "\n")
        
    return

    


def test_most_normal_mnistc(model):
    base_images = np.load("./datasets/MNIST-C/mnist_c/identity/test_images.npy")
    base_labels = np.load("./datasets/MNIST-C/mnist_c/identity/test_labels.npy")
    zigzag_images = np.load("./datasets/MNIST-C/mnist_c/zigzag/test_images.npy")
    zigzag_labels = np.load("./datasets/MNIST-C/mnist_c/zigzag/test_labels.npy")
    
    idx, labels, scores = zip(*model.test_scores)
    
    idx2 = [idx[i] for i in range(len(scores)) if labels[i]==0]
    scores2 = [scores[i] for i in range(len(scores)) if labels[i]==0]
    
    
    ind = np.argsort(scores2)
    
    minTest = ind[:5]  
    
    #img,_,_,_ = model.dataset.test_set.__getitem__(idx2[minTest[i]])

    
    for i in range(10):
        ind = idx2[minTest[i]]
        plot_explaination(model, base_images[ind].reshape((28,28))/255, zigzag_images[ind].reshape((28,28))/255)
    


    
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
        
    
    
    

def main():
    test_cosine_mnist( None, anomaly="translate")
    
    #path="./saved_models_DSVDD/DSVDD_MNIST_1_normal_random.model"
    #path="./saved_models_DSAD/DSAD_MNIST_8_hypersphere.model"
    #path="./saved_models_DSVDD/DSVDD_MVTec_1_normal_none.model"
    
    #model = pipe(c=0, do_pretrain=False,net_name="mnist_LeNet", dataset_name="mnist", do_print=True, epochs=10, loss="normal")
    
    model = pipe(c=14, do_pretrain=False,net_name="mvtec_vgg", dataset_name="mvtec", do_print=True, epochs=10, loss="normal")
    test_cosine_mvtec(model, normal_class=model.nc, cosine=True)
    test_cosine_mvtec(model, normal_class=model.nc, cosine=False)
    del model
    gc.collect()
    
    model = pipe(c=0, do_pretrain=False,net_name="mvtec_vgg", dataset_name="mvtec", do_print=True, epochs=10, loss="hypersphere")
    test_cosine_mvtec(model, normal_class=model.nc, cosine=True)
    test_cosine_mvtec(model, normal_class=model.nc, cosine=False)
    del model
    gc.collect()
    
    model = pipe(c=0, do_pretrain=False,net_name="mvtec_vgg", dataset_name="mvtec", do_print=True, epochs=10, loss="bce")
    test_cosine_mvtec(model, normal_class=model.nc, cosine=True)
    test_cosine_mvtec(model, normal_class=model.nc, cosine=False)
    del model
    gc.collect()

    
    return





    
    paths_mnist_c = ["./saved_models_DSVDD/DSVDD_MNIST_-1_normal_oe_emnist.model",
                        "./saved_models_DSVDD/DSVDD_MNIST_-1_normal_random.model",
                        "./saved_models_DSAD/DSAD_MNIST_-1_normal_none.model",
                        "./saved_models_DSAD/DSAD_MNIST_-1_normal_all.model",
                        "./saved_models_DWSAD/DWSAD_MNIST_-1_normal_none.model"]
    
    paths_weak_sup = ["./saved_models_DWSAD/DWSAD-16_MNIST_-1_normal_random.model",
                      "./saved_models_DWSAD/DWSAD-16384_MNIST_-1_normal_random.model"]


    paths = paths_weak_sup
    for i in range(len(paths)):
        model = load_model(paths[i])
        print(paths[i])
        folders = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]
        
        for f in folders:
            model.dataset.load_new_test_set([f])
            model.test()
            print(f, "AUC:",model.test_auc)
            test_cosine_mnist(model,anomaly=f, cosine=True)
        
        print("\n\n\n")
        
        for f in folders:
            model.dataset.load_new_test_set([f])
            model.test()
            print(f, "AUC:",model.test_auc)
            test_cosine_mnist(model,anomaly=f, cosine=False)
            
        print("\n\n\n")
        print("\n\n\nNEW PATH\n")
    
    #model.test()
    #print("AUC:",model.test_auc)
    
    
    
    #for i in range(10, 15):
    #    path="./saved_models_DSVDD/DSVDD_MVTec_{}_normal_none.model".format(i)
    #    model = load_model(path)
    #    model.test()
    #    print(i, "AUC:",model.test_auc)
    #    if model.nc != i :
    #        print("kaggear")
    #    test_cosine_mvtec(model, normal_class=model.nc)
    #    
    #    del model.net
    #    del model
    #    torch.cuda.empty_cache()
     
     
    
    #model = pipe(c=-1)
        
        
    #test_cosine_mvtec(model, normal_class=model.nc)
    


    
    
    

if __name__ == '__main__':
    main()
    
    
"""
Weak-DSAD no additional noise (10 supervised samples):
AUC:0.9999978279999998
Explain-Acc:0.34973666666666664
Clever Hans:0.6502611613333333

DSAD with all noises combined (random+emnist):
AUC:0.9999672993333333
Explain-Acc:0.33399200426666664
Clever Hans:0.6659752939453913

DSAD without any noise:
AUC:1.0
Explain-Acc:0.20462614393333334
Clever Hans:0.7953738565246264

DSVDD with random noise:
AUC:0.9235965693333335
Explain-Acc:0.2939470804353333
Clever Hans:0.6296494899805317

DSVDD with EMNIST OE:
AUC:0.9921832359999999
Explain-Acc:0.4696777733333333
Clever Hans:0.5225054668556238 (edited) 


"""
