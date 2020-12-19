from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#from noise import Noise_Dataset


base_images = np.load("./MNIST-C/mnist_c/identity/test_images.npy")
base_labels = np.load("./MNIST-C/mnist_c/identity/test_labels.npy")

zigzag_images = np.load("./MNIST-C/mnist_c/zigzag/test_images.npy")
zigzag_labels = np.load("./MNIST-C/mnist_c/zigzag/test_labels.npy")

print(np.max(base_images[0]))


def mnist_c_test(k=5):
    fig, ax = plt.subplots(k,5)
    for i in range(k):
        ax[0,i].set_title(str(base_labels[i]))
        ax[0,i].imshow(base_images[i].reshape((28,28)), cmap="gray_r")
        
        ax[1,i].set_title(str(zigzag_labels[i]))
        ax[1,i].imshow(zigzag_images[i].reshape((28,28)), cmap="gray_r")
        

        ax[2,i].imshow(zigzag_images[i].reshape((28,28))-base_images[i].reshape((28,28)), cmap="gray_r")

    plt.tight_layout()
    plt.show()
    
    return

def test_most_normal(model, k=5):
    idx, labels, scores = zip(*model.test_scores)
    
    idx2 = [idx[i] for i in range(len(scores)) if labels[i]==0]
    scores2 = [scores[i] for i in range(len(scores)) if labels[i]==0]
    
    
    ind = np.argsort(scores2)
    
    minTest = ind[:5]
    return
    
    
def plot_images_overview(k):
    #without identity
    folders = ["brightness","fog","impulse_noise","scale","spatter","zigzag","canny_edges","glass_blur","motion_blur","shear","stripe","dotted_line","rotate","shot_noise","translate"]

    fig, ax = plt.subplots(3,5)
    
    for i in range(15):
        a,b = divmod(i, 5)
        print(i, b,a)
        set_images = np.load("./MNIST-C/mnist_c/{}/test_images.npy".format(folders[i]))
        set_images = np.squeeze(set_images)
        print(set_images.shape)
        ax[a,b].set_title(folders[i])
        ax[a,b].imshow(set_images[k].reshape((28,28)),vmin=0,vmax=255, cmap="gray")
        ax[a,b].axis("off")
        
    plt.tight_layout()
    plt.show()
    
    
def plot_noise():
    fig, ax = plt.subplots(2,3)
    
    noise1 = Noise_Dataset(noise=("gaussian",), size=1000, image_size=(1,28,28))
    noise2 = Noise_Dataset(noise=("blobs",), size=1000, image_size=(1,28,28))
    noise3 = Noise_Dataset(noise=("bernoulli",), size=1000, image_size=(1,28,28))
    
    ax[0,0].imshow(noise1.train_set.__getitem__(0), cmap="gray")
    ax[0,1].imshow(noise2.train_set.__getitem__(0), cmap="gray")
    ax[0,2].imshow(noise3.train_set.__getitem__(0), cmap="gray")
    
    ax[1,0].imshow(noise1.train_set.__getitem__(0), cmap="gray")
    ax[1,1].imshow(noise2.train_set.__getitem__(0), cmap="gray")
    ax[1,2].imshow(noise3.train_set.__getitem__(0), cmap="gray")
    
    plt.tight_layout()
    plt.show()
    
        
plot_images_overview(120)
#plot_noise()

    
    
