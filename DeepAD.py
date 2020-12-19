import time
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from networks.main import build_network, build_autoencoder
from datasets.main import load_dataset
from datasets.weak_sup import weak_supervision_Dataset


class DeepAD(object):

    def __init__(self, method, net_name, d_name, epochs=10, lr=0.001, nc=4, lr_milestones=[100], loss="normal", do_print=True, do_pretrain=True, pretrain_epochs=None, noise=None, weak_supervision=False, weak_supervision_size=10, device="cpu", weak_supervision_set="mnistc", out_file=None):
        
        
        self.eta = 1.0
        self.eps = 1e-6
        
        self.nc = nc
        self.loss = loss
        self.do_print= do_print
        self.do_pretrain = do_pretrain
        self.out_file=out_file
        
        if d_name == "mvtec":
            self.do_pretrain=False
        
        self.method = method
        self.dataset_name = d_name
        
        self.noise = noise
        
        if self.method == "DSVDD":
            self.dataset = load_dataset(d_name, "./datasets/", normal_class=self.nc, n_known_outlier_classes=5, known_outlier_class=2 , ratio_known_outlier=0.0 , ratio_known_normal=0.0, ratio_pollution=0.0, noise=self.noise)
        else:
            #self.dataset = load_dataset(d_name, "./datasets/", normal_class=self.nc, n_known_outlier_classes=5, known_outlier_class=2 , ratio_known_outlier=0.5 , ratio_known_normal=0.5, ratio_pollution=0.1, noise=None)
            self.dataset = load_dataset(d_name, "./datasets/", normal_class=self.nc, n_known_outlier_classes=9, known_outlier_class=2 , ratio_known_outlier=0.1 , ratio_known_normal=0.1, ratio_pollution=0.0, noise=None)
            
        self.weak_supervision_size = weak_supervision_size
        self.supervision_dataset = weak_supervision_Dataset(root="./datasets/", size=self.weak_supervision_size, dataset=weak_supervision_set, nc=self.nc)
        self.weak_supervision = weak_supervision
        
        self.net_name = net_name
        self.net = build_network(net_name, loss=self.loss)
        
        self.n_epochs = epochs
        self.lr = lr
        self.lr_milestones = lr_milestones
        self.optimizer = "adam"
        self.weight_decay = 1e-6
        self.batch_size = 128
        
        if d_name == "mvtec":
            self.batch_size = 32
        
        if pretrain_epochs==None:
            self.pretrain_epochs = self.n_epochs
        else:
            self.pretrain_epochs = pretrain_epochs
        
        
        self.device=device
        
        if do_print:
            print("SETTINGS: eta:{}; eps:{}; weight_decay:{}; normal_class:{}; loss:{}; method:{}; dataset:{}; epochs:{}; lr:{}; pretrain:{}; pretrain_epochs:{}; noise:{}; weakSup:{}; weakSupSize:{}\n\n".format(self.eta, self.eps, self.weight_decay, self.nc, self.loss, self.method, self.dataset_name, self.n_epochs, self.lr, self.do_pretrain, self.pretrain_epochs, self.noise, self.weak_supervision, self.weak_supervision_size), file=self.out_file)
        
        if self.do_pretrain:
            self.pretrain()
            self.init_network_weights_from_pretraining()
            self.set_network()
        else:
            self.c = torch.rand(self.net.rep_dim, device=self.device)


    def train(self):
        net = self.net
        net = net.to(self.device)
        
        if self.weak_supervision:
            supervision_size = min(self.weak_supervision_size, 10)
            train_loader, _ = self.dataset.loaders(batch_size=self.batch_size-supervision_size, num_workers=1, shuffle_train=True)
            supervision_loader, _ = self.supervision_dataset.loaders(batch_size=supervision_size, num_workers=1, shuffle_train=True)
        else:
            train_loader, _ = self.dataset.loaders(batch_size=self.batch_size, num_workers=1, shuffle_train=True)
        
        #c = torch.tensor([[random.random() for j in range(28)] for i in range(28)])
        #self.c = net(c)
        
        train_start_time = time.time()
        
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        
        for epoch in range(self.n_epochs):
            
            if self.weak_supervision:
                supervision_loader_iter = iter(supervision_loader)
            
            if epoch%5==0 and self.do_print:
                self.test()
                print("AUC:", self.test_auc, file=self.out_file)

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            semi_targets_sum = 0
            
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                #plt.switch_backend("TKAgg")
                #for i in inputs:
                    #plt.imshow(i.permute(1,2,0))
                    #plt.show()
                
                #print(inputs.shape)
                
                if self.weak_supervision:
                    try:
                        supervised_inputs, _, supervised_targets, _ = next(supervision_loader_iter)  
                    except StopIteration:
                        supervision_loader_iter = iter(supervision_loader)
                        supervised_inputs, _, supervised_targets, _ = next(supervision_loader_iter)
                                        
                    supervised_inputs, supervised_targets = supervised_inputs.to(self.device), supervised_targets.to(self.device)
                    inputs = torch.cat([inputs, supervised_inputs])
                    semi_targets = torch.cat([semi_targets, supervised_targets])
                                    

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                
                
                #0 -> normal
                #1 -> anomaly
                #if self.loss != "bce":
                dist = torch.sum((outputs) ** 2, dim=1)
                    
                a = abs(sum(semi_targets)).float()
                b = float(len(semi_targets))
                eta_anomal =  (b/2)/a
                eta_normal = (b/2)/(b-a)
            
                if a == 0:
                    eta_normal=1
                    eta_anomal=0
                
                if self.loss == "hypersphere":
                    losses = torch.where(semi_targets == 0, eta_normal* dist, eta_anomal*(-1)*torch.log(1 - torch.exp((-1)* dist)+ self.eps))
                    loss = torch.mean(losses)
                    
                elif self.loss =="normal":
                    losses = torch.where(semi_targets == 0, eta_normal* dist, eta_anomal * ((dist + self.eps) ** (semi_targets.float())))
                    loss = torch.mean(losses)
                    
                elif self.loss =="bce":
                    net.l4.requires_grad = True
                    sig = nn.Sigmoid()
                    out = net.l4(outputs)
                    targets = (semi_targets*(-1)).float()
                    dist = sig(out)
                    dist = dist.view(-1)
                    weights = (targets*(eta_anomal-eta_normal))+eta_normal
                    bce = nn.BCELoss(weight=weights, reduction='none')

                    losses = bce(dist, targets)
                    loss = torch.mean(losses)
                
                else:
                    print("no implemented loss function specified")
                    
                    
                semi_targets_sum += torch.sum(semi_targets)
                epoch_loss += loss.item()
                
                loss.backward()#retain_graph=True)
                optimizer.step()

                n_batches += 1

            scheduler.step()
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            if self.do_print:
                print("Training: ", "{:.2f}%, Time {:.2f}s, Loss {:.3f}".format(100*(epoch + 1)/self.n_epochs, epoch_train_time, epoch_loss / n_batches), file=self.out_file)
                #print(semi_targets_sum)
            
        self.train_time = time.time() - train_start_time
        self.net = net


    def test(self):
        net = self.net
        net = net.to(self.device)
        
        _, test_loader = self.dataset.loaders(batch_size=self.batch_size, num_workers=1)
        
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        
        
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs) ** 2, dim=1)
                
                if self.loss =="bce": 
                    sig = nn.Sigmoid()
                    out = net.l4(outputs)
                    dist = sig(out)
                    dist = dist.view(-1)
                
                
                '''
                if self.loss == "hypersphere":
                    dist = torch.sum((outputs) ** 2, dim=1)
                else:
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                '''
                   
                loss = torch.mean(dist)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        
        self.mean_anomaly_score = np.mean(scores)
        self.percentile = np.percentile(scores, 80)
        
        self.test_auc = roc_auc_score(labels, scores)
        
    
    def save_results(self, export_path):
        f = open(export_path, "w")
        f.write("parameter:")
        params = [self.method, self.net_name, self.n_epochs, self.lr, self.optimizer, self.weight_decay, self.batch_size]
        
        for p in params:
            f.write("{}\n".format(p))
            
        f.write("\n")
        f.write("Traintime: {}".format(self.train_time))
        f.write("Testtime: {}".format(self.test_time))
        f.write("\nAUC: {}".format(self.test_auc))
        f.close
        
    
    def save_model(self, export_path):
        torch.save(self.net.state_dict(), export_path)
        return
        
    
    def load_model(self, import_path):
        self.net.load_state_dict(torch.load(import_path))
        return
    
    
    def init_center_c(self, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data. And safe the mean of all inputs as the baseline"""
        
        net = self.net
        train_loader, _ = self.dataset.loaders(batch_size=self.batch_size, num_workers=1)
        
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        I = []

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                I.append(inputs)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples
        I = torch.stack(I)
        I = torch.mean(I)
        
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c, I


    def set_network(self):
        self.c, self.baseline = self.init_center_c()


    def pretrain(self):
        self.ae_net = build_autoencoder(self.net_name)
        
        ae_net = self.ae_net
        
        criterion = nn.MSELoss(reduction='none')
        
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)
        
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        train_loader, test_loader = self.dataset.loaders(batch_size=self.batch_size, num_workers=1, shuffle_train=True)
        
        start_time = time.time()
        ae_net.train()
        
        for epoch in range(self.pretrain_epochs):
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            if self.do_print:
                print("Pretraining: ", "{:.2f}%, Time {:.2f}s, Loss {:.3f}".format(100*(epoch + 1)/self.pretrain_epochs, epoch_train_time, epoch_loss / n_batches), file=self.out_file)

                
        self.ae_train_time = time.time() - start_time
        self.ae_net = ae_net
     
     
    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)
        
        
    def deletion_test(self):
        net = self.net
        _, test_loader = self.dataset.loaders(batch_size=self.batch_size, num_workers=1)

        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()

        for data in test_loader:
            inputs, labels, semi_targets, idx = data
            inputs2 = []

            for i in range(len(inputs)):

                
                img = torch.tensor(inputs[i].cpu().detach().numpy(), requires_grad=True)
                
                R = LRP(self, img)
                r = R[0].cpu().detach().numpy()[0]
                
                img2 = pixel_flipping(inputs[i,0].cpu().detach().numpy(), r, amount=70)
                inputs2.append(img2)



            inputs = np.array(inputs2)
            inputs = torch.as_tensor(inputs).float()

            outputs = net(inputs)
            dist = torch.sum((outputs - self.c) ** 2, dim=1)
            loss = torch.mean(dist)
            scores = dist

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                        labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

            epoch_loss += loss.item()
            n_batches += 1

        self.px_test_time = time.time() - start_time
        self.px_test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.px_test_auc = roc_auc_score(labels, scores)
        
        
    
        
    
