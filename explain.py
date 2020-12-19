import torch
import numpy as np
import utils.utils as utils


def LRP(model, x):
	# reshape input
	#1. forward pass
	#2. compute loss
	#3. backward pass

    if model.dataset_name == "cifar10":
        x = x.view(-1, 3, 32, 32)
    elif model.dataset_name =="mvtec":
        x = x.view(-1, 3, 224, 224)
    else:
        x = x.view(-1, 1, 28, 28)
        
    
    layers = list(model.net.modules())[1:]
    layers2 = []

    for i1 in range(len(layers)):
        if isinstance(layers[i1], torch.nn.modules.container.Sequential):
            for i2 in layers[i1]:
                layers2.append(i2)
    
    layers = layers2
    L = len(layers)
    
    
    old_model = True
    
    #if model.loss != "bce":
    if old_model:
        L += 1
    
    A = [x]+[None]*L
    for l in range(L-1): 
        if isinstance(layers[l], torch.nn.Linear):
            A[l+1] = layers[l].forward(A[l].view(int(A[l].size(0)), -1))
        else:
            A[l+1] = layers[l].forward(A[l])
        
            
    if model.loss != "bce":  
        if old_model:
            layers.append("dist")
        else:
            layers[-1] = "dist"
        loss = torch.sum((A[-2]) ** 2, dim=1)
        A[-1] = (A[-2])**2
    else:
        sig = torch.nn.Sigmoid()
        out = model.net.l4.forward(A[-2])
        layers.append(model.net.l4)
        A[-1] = sig(out)
        
        
    R = [None]*L + [(A[-1]).data]
        
    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        
        
        if isinstance(layers[l],torch.nn.MaxPool2d): 
            layers[l] = torch.nn.AvgPool2d(2)
        if layers[l] == "dist":
            R[l] = R[l+1]

        
        elif isinstance(layers[l], torch.nn.Linear):
            rho = lambda p: p
            incr = lambda z: z+1e-6+0.25*((z**2).mean()**.5).data
            
            q = A[l].view(int(A[l].size(0)), -1)
            z = incr(utils.newlayer(layers[l], rho).forward(q))
            s = (R[l+1]/z).data  
            
            (z*s).sum().backward(); 
            c = A[l].grad 
            R[l] = (A[l]*c).data 

        elif isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
            rho = lambda p: p.clamp(min=0)                      
            incr = lambda z: z+1e-9
            
            z = incr(utils.newlayer(layers[l], rho).forward(A[l]))
            s = (R[l+1]/z).data 
            (z*s).sum().backward(); 
            c = A[l].grad 
            R[l] = (A[l]*c).data                           
        else:
            R[l] = R[l+1]
            
            
            
    
    A[0] = (A[0].data).requires_grad_(True)
    

    if model.dataset_name=="mvtec":
        mean = 0
        std  = 1
        
        lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
        hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)
    else:
        
        lb = (A[0].data*0+(0)).requires_grad_(True)
        hb = (A[0].data*0+(1)).requires_grad_(True)

    z = layers[0].forward(A[0]) 
    z -= utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)   
    z -= utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)   
    z += 1e-6
    s = (R[1]/z).data                                                   
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad        
    R[0] = (A[0]*c+lb*cp+hb*cm).data  

            
    return R




#computes gradient, gradient*input and saliency map for given input
def grads(model, input):
    input.requires_grad_(True)
    forward = model.net(input)
    
    #compute loss
    dist = torch.sum((forward - model.c) ** 2, dim=1)
    loss = torch.mean(dist)
    loss.backward(retain_graph=True)
    
    i = input.detach().numpy()
    g = input.grad
    
    return input.detach().numpy(), g, g*i , g**2 


def integrated_gradient(model, input, b=0):
	#baseline is nearly white picture with small eps in every pixel (or completely black baseline)
	
    input.requires_grad_(True)
    if b==0:
        baseline = torch.ones(input.shape) * 0.0001
    if b==1:
        baseline = torch.zeros(input.shape)
    if b==2:
        baseline = model.baseline

    steps = 50
    scaled_inputs = [baseline + (float(i) / steps) * (input - baseline) for i in range(0, steps + 1)]
    grads = []

    for si in scaled_inputs:
        outputs = model.net(si)  
        
        dist = torch.sum((outputs - model.c) ** 2, dim=1)
        loss = torch.mean(dist)
            
        loss.backward(retain_graph=True)
        grads.append(input.grad.cpu().detach().numpy())
        
    grads = np.array(grads)
    avg_grads = np.average(grads[:-1], axis=0)
    integrated_grad = (input.detach().cpu().detach().numpy() - baseline.cpu().detach().numpy()) * avg_grads

        
    return integrated_grad




def top_k_similarity(explaination, groundtruth, k=30):
    
    
    X = np.argsort(explaination[0])
    k = min(k, int(torch.sum(groundtruth).data))
    
    sim = 0
    
    for i in range(k):
        x = X[-i-1]
        if groundtruth[0,x] != 0:
            sim += 1
    
    return sim/k
    


    
    
