import torch
import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt


def LRP(model, x):
	# reshape input
	#1. forward pass
	#2. compute loss
	#3. backward pass

    if model.dataset_name == "cifar10":
        x = x.view(-1, 3, 32, 32)
    else:
        x = x.view(-1, 1, 28, 28)

    layers = list(model.net.modules())[1:]
    layers2 = []

    for i1 in range(len(layers)):
        if isinstance(layers[i1], torch.nn.modules.container.Sequential):
            for i2 in layers[i1]:
                layers2.append(i2)
    #print(len(layers), layers)
    
    layers = layers2
    L = len(layers)+1
    
    A = [x]+[None]*L
    for l in range(L-1): 
        if isinstance(layers[l], torch.nn.Linear):
            A[l+1] = layers[l].forward(A[l].view(int(A[l].size(0)), -1))
        else:
            A[l+1] = layers[l].forward(A[l])
            
    layers.append("dist")
            
    
    loss = torch.sum((A[-2] - model.c) ** 2, dim=1)
    #print(loss)
    
    #A[-1] = (A[-2] - model.c)**2
    A[-1] = (A[-2])**2
    
    #T = torch.FloatTensor((1.0*(np.arange(1)==1).reshape([1,1,1,1])))
    #R = [None]*L + [(A[-1]*T).data]
    #
    R = [None]*L + [(A[-1]).data]
    
    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        #A[l].requires_grad_(True)
        
        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
        
        if layers[l] == "dist":
            R[l] = R[l+1]
            #i = torch.argmax(R[l+1])
            #R[l]  = torch.zeros(R[l+1].shape)
            #R[l][0,i] = loss
            
        
        elif isinstance(layers[l], torch.nn.Linear):
            rho = lambda p: p
            incr = lambda z: z+1e-9
            
            
            q = A[l].view(int(A[l].size(0)), -1)
            f = utils.newlayer(layers[l],rho).forward(q)
            f2 = layers[l].forward(q)
            z = incr(f2)  # step 1

            s = (R[l+1]/z).data            # step 2
            
            (z*s).sum().backward(create_graph=True); 
            c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data 

        elif isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):

            #if l <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
            #if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            #if l >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

            rho = lambda p: p                       
            incr = lambda z: z+1e-9
            
            z = incr(utils.newlayer(layers[l],rho).forward(A[l]))  # step 1
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward(); 
            c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data                                   # step 4
            
        else:
            
            R[l] = R[l+1]
            
    
    A[0] = (A[0].data).requires_grad_(True)

    lb = (A[0].data*0+0).requires_grad_(True)
    hb = (A[0].data*0+1).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    z -= utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data      
    
            
    return R


def LRP_a1b0(model, x):
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
    
    #print(L)
    
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
        
        
    #print("\n")
    #print(loss)
    
    #T = torch.FloatTensor((1.0*(np.arange(1)==1).reshape([1,1,1,1])))
    #R = [None]*L + [(A[-1]*T).data]
    #
    R = [None]*L + [(A[-1]).data]
        
    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        
        #A[l].requires_grad_(True)
        
        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
        
        if layers[l] == "dist":
            R[l] = R[l+1]
            #i = torch.argmax(R[l+1])
            #R[l]  = torch.zeros(R[l+1].shape)
            #R[l][0,i] = loss
            
        
        elif isinstance(layers[l], torch.nn.Linear):
            #print("Lin")
            #rho = lambda p: p.clamp(min=0)
            rho = lambda p: p
            incr = lambda z: z+1e-6+0.25*((z**2).mean()**.5).data
            
            q = A[l].view(int(A[l].size(0)), -1)
            z = incr(utils.newlayer(layers[l], rho).forward(q))
            s = (R[l+1]/z).data  
            
            (z*s).sum().backward(); 
            c = A[l].grad 
            #print(torch.sum(torch.isnan(c)))
            R[l] = (A[l]*c).data 
            #print(torch.sum(torch.isnan(R[l])))
            #print(torch.sum(torch.isnan(A[l])))
            #print("\n")
            

        elif isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):

       
            rho = lambda p: p.clamp(min=0)                      
            incr = lambda z: z+1e-9
            
            z = incr(utils.newlayer(layers[l], rho).forward(A[l]))
            #print(layers[l])
            #print(torch.sum(torch.isnan(R[l+1])))
            #print(torch.sum(torch.isnan(z)))
            
            s = (R[l+1]/z).data 
            
            #print(torch.sum(torch.isnan(s)))
            #print("\n")
            
            (z*s).sum().backward(); 
            c = A[l].grad 
            R[l] = (A[l]*c).data                                # step 4
            
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
    #print("go")
    #print(torch.sum(torch.isnan(A[0])))
    #print(torch.sum(torch.isnan(lb)))
    #print(torch.sum(torch.isnan(hb)))


    z = layers[0].forward(A[0]) # step 1 (a)
    z -= utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    z -= utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    
    z += 1e-6
    
    
    #print(torch.sum(torch.isnan(R[1])))
    #print(torch.sum(torch.isnan(z)))
    
    #for k in z:
    #    for j in k:
    #        for o in j:
    #            for p in o:
    #                if p == 0:
    #                    p = 1e-9
    #                    print("shiets")
    
    s = (R[1]/z).data                                                      # step 2
    
    #print(torch.sum(torch.isnan(s)))
    #print("\n\n")
    
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    
    #print(torch.sum(torch.isnan(s)))

    
    R[0] = (A[0]*c+lb*cp+hb*cm).data  
    
    #print(torch.sum(torch.isnan(R[1])))
    #print(torch.sum(torch.isnan(R[0])))
            
    return R



def deep_taylor_decomp(model, input):
    pass
	
	
def shapley(model, input):

	
	
	for i in range(input.shape[0]):
		for j in range(input.shape[1]):
			if input[i,j] == 0:
				used[i,j] = 1
				shap[i,j] = 0
				
	for i in range(input.shape[0]):
		for j in range(input.shape[1]):			
			shap[i,j] = shapley_recursion(model, input, used, np.sum(used)+10)[i,j]
			
	return shap
	
	

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


#delete the most "relevant" pixel
def pixel_flipping(input, R, amount=70):
    #only destroys pixels
	#input numpy array
    
    X, Y = np.unravel_index(np.argsort(R, axis=None), input.shape)
    
    transformed_inputs = input.copy()
    
    for i in range(amount):
        x,y = X[-i-1], Y[-i-1]
        transformed_inputs[x,y] = 0
    
    return transformed_inputs


def plot_explaination(model, original, anomaly=None):
    if anomaly is None:
        anomaly = original
        
    
        
    cm = "seismic_r"
    fig, ax = plt.subplots(2,6)
    
    #Compute explainations ----------------------------------------------
    
    original = torch.tensor(original).float()
    anomaly = torch.tensor(anomaly).float()

        
    org_out = model.net(original)
    org_dist = torch.sum((org_out - model.c) ** 2, dim=1)
    
    ano_out = model.net(anomaly)
    ano_dist = torch.sum((ano_out - model.c) ** 2, dim=1)
    
    
    iG1 = integrated_gradient(model, anomaly)
    iG2 = integrated_gradient(model, anomaly, b=1)
    iG3 = integrated_gradient(model, anomaly, b=2)


    R = LRP(model, anomaly)
    r = R[0].cpu().detach().numpy()[0]
    
    R2 = LRP_a1b0(model, anomaly)
    r2 = R2[0].cpu().detach().numpy()[0]
    
    img3, g1,g2,g3 = grads(model, anomaly)
    
    #Plot original + anomaly --------------------------------------------
    
    ax[0,0].set_title("Original\n Score={:.3f}".format(org_dist[0]))
    ax[0,0].imshow(original.cpu().detach().numpy(), cmap="gray_r")
    
    ax[0,1].set_title("Modified\n Score={:.3f}".format(ano_dist[0]))
    ax[0,1].imshow(anomaly.cpu().detach().numpy(), cmap="gray_r")
    
    
    #Plot explainations -------------------------------------------------
    
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
    b = [np.max(iG1), -1*np.min(iG1)]
    ax[1,2].imshow(iG1, cmap=cm, vmin=-1*max(b), vmax=max(b))
    
    ax[1,3].set_title("LRP")
    b = [np.max(r[0]), -1*np.min(r[0])]
    ax[1,3].imshow(r[0], cmap="seismic_r", vmin=-1*max(b), vmax=max(b))
    
    ax[0,5].set_title("LRP_a1b0")
    b = [np.max(r2[0]), -1*np.min(r2[0])]
    print(b, sum(sum(r2[0])))
    ax[0,5].imshow(r2[0], cmap="seismic_r", vmin=-1*max(b), vmax=max(b))
    
    ax[0,4].set_title("iG_2")
    b = [np.max(iG2), -1*np.min(iG2)]
    ax[0,4].imshow(np.abs(iG2), cmap=cm, vmin=-1*max(b), vmax=max(b))
    
    ax[1,4].set_title("iG_3")
    b = [np.max(iG3), -1*np.min(iG3)]
    ax[1,4].imshow(np.abs(iG3), cmap=cm, vmin=-1*max(b), vmax=max(b))
    
            
    ax[1,0].axis("off")
    ax[1,5].axis("off")
    
    
    plt.tight_layout()
    plt.show()
    
    return


def top_k_similarity(explaination, groundtruth, k=30):
    
    
    X = np.argsort(explaination[0])
    k = min(k, int(torch.sum(groundtruth).data))
    
    sim = 0
    
    for i in range(k):
        x = X[-i-1]
        if groundtruth[0,x] != 0:
            sim += 1
    
    return sim/k
    


    
    
