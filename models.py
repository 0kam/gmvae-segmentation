from distributions import Inference2D, Generator2D, Classifier2D, Prior2D
from utils import TrainDS2D, TestDS2D
from pixyz.losses import ELBO
from pixyz.models import Model
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from glob import glob

class GMVAE2D_US:
    """
    GMVAE object for unsupervised 2D (single) image segmentation.
    """
    def __init__(self, data_dir, kernel_size=(9,9), num_cluster = 30, z_dim=64, batch_size=500, device = "cuda"):
        """
        Parameters
        ----------
        data_dir : str
            A data directory.
        kernel_size : tuple of int default (9, 9)
            A kernel size.
        num_cluster : int
            Number of Gaussian mixture
        z_dim : int default 64
            A latent variable dimension
        batch_size : int default 500
            A batch size
        device : str default "cuda"
            A device to use with pytorch.
        """
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.num_cluster = num_cluster
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.device = device

        # set a model
        x_dim = self.kernel_size[0] * self.kernel_size[1] * 3
        
        self.p = Generator2D(x_dim, z_dim).to(device)
        self._q = Inference2D(x_dim, z_dim, y_dim=self.num_cluster).to(device)
        self.f = Classifier2D(x_dim, self.num_cluster).to(device)
        self.q = self._q * self.f
        self.prior = Prior2D(z_dim, y_dim=self.num_cluster).to(device)
        self.p_joint = self.p * self.prior
            
        elbo = ELBO(self.p_joint, self.q)
        loss_cls = -elbo.mean()
        
        self.model = Model(loss_cls,test_loss=loss_cls,
                      distributions=[self.p, self._q, self.f, self.prior], 
                      optimizer=optim.Adam, optimizer_params={"lr":1e-3})
        print(self.model)

        dataset = TrainDS2D(self.data_dir, self.kernel_size)
        n_samples = len(dataset)
        train_size = int(n_samples * 0.8)
        
        subset1_indices = list(range(0, train_size))
        subset2_indices = list(range(train_size, n_samples))
        
        train_dataset = Subset(dataset, subset1_indices)
        val_dataset   = Subset(dataset, subset2_indices)
        
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle = False, num_workers=8)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle = False, num_workers=8)

    def _train(self, epoch):
        train_loss = 0
        for x in tqdm(self.train_loader):
            x = x.to(self.device)
            loss = self.model.train({"x": x})
            train_loss += loss
            
        train_loss = train_loss
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        
        return train_loss
    
    def _val(self, epoch):
        val_loss = 0
        for x in self.val_loader:
            x = x.to(self.device)       
            loss = self.model.test({"x": x})
            val_loss += loss
            
            pred_y = self.f.sample_mean({"x": x}).argmax(1).detach().cpu().numpy()

        print('Test loss: {:.4f}'.format(val_loss))
        return val_loss
    
    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch)
            val_loss = self._val(epoch)
      
    def draw(self, image_path, out_path):
        with Image.open(image_path) as img:
            w, h = img.size
        dataset = TestDS2D(image_path, self.kernel_size)
        loader = DataLoader(dataset, self.batch_size, shuffle=False, num_workers=0)
        pred_ys = []
        #u_coords = []
        #v_coords = []
        for x, ind in loader:
            x = x.to(self.device)
            pred_y = self.f.sample_mean({"x": x}).argmax(1).detach().cpu()
            pred_ys.append(pred_y)
            #u = ind[0]
            #v = ind[1]
            #u_coords.append(u)
            #v_coords.append(v)
        
        seg_image = torch.cat(pred_ys).reshape([w,h]).numpy()
        cmap = plt.get_cmap("jet", self.num_cluster)
        plt.imsave(out_path, seg_image, cmap = cmap)
            

gmvae = GMVAE2D_US(data_dir="crop/2010", kernel_size=(5,5), num_cluster = 10,
    z_dim=64, batch_size=5000, device="cuda")
gmvae.train(300)


import os
if not os.path.exists("result"):
    os.mkdir("result/2010")
    os.mkdir("result/2020")

for path in glob("crop/2010/*"):
    gmvae.draw(path, path.replace("crop", "result"))



class GMVAE3D_US:
    """
    GMVAE object for unsupervised 3D (time-series) image segmentation.
    """
    def __init__(self, data_dir, kernel_size=(9,9), num_cluster = 30, z_dim=64, batch_size=500, device = "cuda"):
        """
        Parameters
        ----------
        data_dir : str
            A data directory.
        kernel_size : tuple of int default (9, 9)
            A kernel size.
        num_cluster : int
            Number of Gaussian mixture
        z_dim : int default 64
            A latent variable dimension
        batch_size : int default 500
            A batch size
        device : str default "cuda"
            A device to use with pytorch.
        """
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.num_cluster = num_cluster
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.device = device

        # set a model
        x_dim = self.kernel_size[0] * self.kernel_size[1] * 3
        seq_length = len(glob(glob(data_dir + "/*")[0] + "/*"))
        
        self.p = Generator3D(x_dim, z_dim, seq_length, device).to(device)
        self._q = Inference3D(x_dim, z_dim, y_dim=self.num_cluster).to(device)
        self.f = Classifier3D(x_dim, self.num_cluster).to(device)
        self.q = self._q * self.f
        self.prior = Prior3D(z_dim, y_dim=self.num_cluster).to(device)
        self.p_joint = self.p * self.prior
            
        elbo = ELBO(self.p_joint, self.q)
        loss_cls = -elbo.mean()
        
        self.model = Model(loss_cls,test_loss=loss_cls,
                      distributions=[self.p, self._q, self.f, self.prior], 
                      optimizer=optim.Adam, optimizer_params={"lr":1e-3})
        print(self.model)

        dataset = TrainDS3D(self.data_dir, self.kernel_size)
        n_samples = len(dataset)
        train_size = int(n_samples * 0.8)
        
        subset1_indices = list(range(0, train_size))
        subset2_indices = list(range(train_size, n_samples))
        
        train_dataset = Subset(dataset, subset1_indices)
        val_dataset   = Subset(dataset, subset2_indices)
        
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle = False, num_workers=8)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle = False, num_workers=8)

    def _train(self, epoch):
        train_loss = 0
        for x in tqdm(self.train_loader):
            x = x.to(self.device)
            loss = self.model.train({"x": x})
            train_loss += loss
            
        train_loss = train_loss
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        
        return train_loss
    
    def _val(self, epoch):
        val_loss = 0
        for x in self.val_loader:
            x = x.to(self.device)       
            loss = self.model.test({"x": x})
            val_loss += loss
            
            pred_y = self.f.sample_mean({"x": x}).argmax(1).detach().cpu().numpy()

        print('Test loss: {:.4f}'.format(val_loss))
        return val_loss
    
    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch)
            val_loss = self._val(epoch)
    
    def draw(self, image_dir, out_path):
        with Image.open(glob(image_dir+"/*")[0]) as img:
            w, h = img.size
        dataset = TestDS3D(image_dir, self.kernel_size)
        loader = DataLoader(dataset, self.batch_size, shuffle=False, num_workers=0)
        pred_ys = []
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                pred_y = self.f.sample_mean({"x": x}).argmax(1).detach().cpu()
                pred_ys.append(pred_y)
        
        seg_image = torch.cat(pred_ys).reshape([w,h]).numpy()
        cmap = plt.get_cmap("jet", self.num_cluster)
        plt.imsave(out_path, seg_image, cmap = cmap)


gmvae = GMVAE3D_US(data_dir="crop", kernel_size=(5,5), num_cluster = 20,
    z_dim=4, batch_size=5000, device="cuda")
gmvae.train(300)

gmvae.draw("crop/2010", "2010_kernel55_cluster20_ep300.png")