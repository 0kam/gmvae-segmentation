from distributions import Inference2D, Generator2D, Classifier2D, Prior2D, Inference3D, Generator3D, Classifier3D, Prior3D
from utils import TrainDS2D, TestDS2D, TrainDS3D, TestDS3D
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
from pathlib import Path
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split

class GMVAE2D_US:
    """
    GMVAE object for unsupervised 2D (single) image segmentation.
    """
    def __init__(self, data_dir, kernel_size=(9,9), num_cluster = 30, z_dim=64, batch_size=500, device = "cuda", num_workers=1):
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
        num_workers : int default 1
            The number of threads to use with torch.utils.DataLoader.
        """
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.num_cluster = num_cluster
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers

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
        
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle = False, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle = False, num_workers=self.num_workers)

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
            

class GMVAE3D_US:
    """
    GMVAE object for unsupervised 3D (time-series) image segmentation.
    """
    def __init__(self, data_dir, kernel_size=(9,9), num_cluster = 30, z_dim=64, batch_size=500, device = "cuda", num_workers=1):
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
        num_workers : int default 1
            The number of threads to use with torch.utils.DataLoader.
        """
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.num_cluster = num_cluster
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers

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
        
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle = False, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle = False, num_workers=self.num_workers)

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

def load_npy(path):
    return torch.tensor(np.load(path))

class GMVAE2D_SS:
    """
    GMVAE object for semisupervised 2D (single) image segmentation.
    """
    def __init__(self, data_dir, z_dim=64, batch_size=3, device = "cuda", num_workers=1):
        """
        Parameters
        ----------
        data_dir : str
            A data directory.
        z_dim : int default 64
            A latent variable dimension
        batch_size : int default 500
            A batch size
        device : str default "cuda"
            A device to use with pytorch.
        num_workers : int default 1
            The number of threads to use with torch.utils.DataLoader.
        """
        self.data_dir = data_dir
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.classes = [Path(n).name for n in glob(data_dir + "/labelled/*")]

        # set dataloaders
        ## labelled data loader
        labelled = DatasetFolder(data_dir+"/labelled/", load_npy, ".npy")
        train_indices, val_indices = train_test_split(list(range(len(labelled.targets))), test_size=0.2, stratify=labelled.targets)
        train_dataset = torch.utils.data.Subset(labelled, train_indices)
        val_dataset = torch.utils.data.Subset(labelled, val_indices)
        x, y = train_dataset[0]
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        
        ## unlabelled loader
        unlabelled = DatasetFolder(data_dir+"/unlabelled/", load_npy, ".npy")
        self.unlabelled_loader = DataLoader(unlabelled, batch_size, shuffle=False)
        # set a model
        self.x_dim = x.shape[1]
        
        self.p = Generator2D(self.x_dim, z_dim).to(device)
        self.q = Inference2D(self.x_dim, z_dim, y_dim=len(self.classes)).to(device)
        self.f = Classifier2D(self.x_dim, y_dim=len(self.classes)).to(device)
        self.prior = Prior2D(z_dim, y_dim=len(self.classes)).to(device)
        self.p_joint = self.p * self.prior
        
        # distributions for unsupervised learning
        _q_u = self.q.replace_var(x="x_u", y="y_u")
        p_u = self.p.replace_var(x="x_u")
        f_u = self.f.replace_var(x="x_u", y="y_u")
        prior_u = self.prior.replace_var(y="y_u")
        q_u = _q_u * f_u
        p_joint_u = p_u * prior_u
        
        p_joint_u.to(device)
        q_u.to(device)
        f_u.to(device)
        
        elbo_u = ELBO(p_joint_u, q_u)
        elbo = ELBO(self.p_joint, self.q)
        nll = -self.f.log_prob() # or -LogProb(f)
        
        rate = 1 * (len(self.unlabelled_loader) + len(self.train_loader)) / len(self.train_loader)
        
        self.loss_cls = -elbo_u.mean() -elbo.mean() + (rate * nll).mean()
        
        self.model = Model(self.loss_cls,test_loss=nll.mean(),
                      distributions=[self.p, self.q, self.f, self.prior], optimizer=optim.Adam, optimizer_params={"lr":1e-3})
        print("Model:")
        print(self.model)

    def _train(self, epoch):
        train_loss = 0
        labelled_iter = self.train_loader.__iter__()
        for x_u, y_u in tqdm(self.unlabelled_loader):
            try: 
                x, y = labelled_iter.next()
            except StopIteration:
                labelled_iter = self.train_loader.__iter__()
                x, y = labelled_iter.next()
            y = y.view(y.shape[0], 1).repeat(1, x.shape[1]).view(-1).unsqueeze(1)
            y = torch.eye(len(self.classes))[y].to(self.device).squeeze()
            x = x.view([-1, self.x_dim])
            x = x.to(self.device)
            
            x_u = x_u.view([-1, self.x_dim])
            x_u = x_u.to(self.device)
            loss = self.model.train({"x": x, "y": y, "x_u": x_u})
            train_loss += loss
        train_loss = train_loss * self.unlabelled_loader.batch_size / len(self.unlabelled_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss
        
    
    def _val(self, epoch):
        test_loss = 0
        total = [0 for _ in range(len(self.classes))]
        tp = [0 for _ in range(len(self.classes))]
        fp = [0 for _ in range(len(self.classes))]
        for x, _y in self.val_loader:
            _y = _y.view(_y.shape[0], 1).repeat(1, x.shape[1]).view(-1).unsqueeze(1)
            y = torch.eye(len(self.classes))[_y].to(self.device).squeeze()
            x = x.view([-1, self.x_dim])
            x = x.to(self.device)
            loss = self.model.test({"x": x, "y": y})
            test_loss += loss
            pred_y = self.f.sample_mean({"x": x}).argmax(dim=1).unsqueeze(1)
            for c in range(len(self.classes)):
                pred_yc = pred_y[_y==c]
                _yc = _y[pred_y==c]
                total[c] += len(_y[_y==c])
                tp[c] += len(pred_yc[pred_yc==c])
                fp[c] += len(_yc[_yc!=c])
        
        test_loss = test_loss * self.val_loader.batch_size / len(self.val_loader.dataset)
        test_recall = [100 * c / t for c,t in zip(tp, total)]
        test_precision = []
        for _tp,_fp in zip(tp, fp):
            if _tp + _fp == 0:
                test_precision.append(0)
            else:
                test_precision.append(100 * _tp / (_tp + _fp))
        recall = {}
        prec = {}
        i = 0
        for c in range(len(self.classes)):
            recall[c] = test_recall[i]
            prec[c] = test_precision[i]
            i += 1
        print("Test Loss:", str(test_loss), "Test Recall:", str(recall), "Test Precision:", str(prec))
        return test_loss, recall, prec
    
    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch)
            val_loss = self._val(epoch)

gmvae = GMVAE2D_SS("data/semisup/tateyama/", 8, 3)
gmvae._val(0)

gmvae.train(200)

test_loss = 0
for x, _y in gmvae.val_loader:
            _y = _y.view(_y.shape[0], 1).repeat(1, x.shape[1]).view(-1).unsqueeze(1)
            y = torch.eye(len(gmvae.classes))[_y].to(gmvae.device).squeeze()
            x = x.view([-1, gmvae.x_dim])
            x = x.to(gmvae.device)
            loss = gmvae.model.test({"x": x, "y": y})
            test_loss += loss
            pred_y = gmvae.f.sample_mean({"x": x}).argmax(dim=1).unsqueeze(1)
            for c in range(len(gmvae.classes)):
                pred_yc = pred_y[_y==c]
                _yc = _y[pred_y==c]
                total[c] += len(_y[_y==c])
                tp[c] += len(pred_yc[pred_yc==c])
                fp[c] += len(_yc[_yc!=c])