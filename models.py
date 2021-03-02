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

data_dir = "./crop/2010"
image_path = "./crop/2010/mrd_085_eos_vis_20100915_1200.jpg"
kernel_size = (5,5)
num_cluster = 30
z_dim = 64
batch_size = 10000
device = "cuda"

# set a model
x_dim = kernel_size[0] * kernel_size[1] * 3

p = Generator2D(x_dim, z_dim).to(device)
_q = Inference2D(x_dim, z_dim, y_dim=num_cluster).to(device)
f = Classifier2D(x_dim, num_cluster).to(device)
q = _q * f
prior = Prior2D(z_dim, y_dim=num_cluster).to(device)
p_joint = p * prior
    
elbo = ELBO(p_joint, q)
loss_cls = -elbo.mean()

model = Model(loss_cls,test_loss=loss_cls,
              distributions=[p, _q, f, prior], 
              optimizer=optim.Adam, optimizer_params={"lr":1e-3})
print(model) 

dataset = TrainDS2D(data_dir, kernel_size)
n_samples = len(dataset)
train_size = int(n_samples * 0.8)

subset1_indices = list(range(0, train_size))
subset2_indices = list(range(train_size, n_samples))

train_dataset = Subset(dataset, subset1_indices)
val_dataset   = Subset(dataset, subset2_indices)

train_loader = DataLoader(train_dataset, batch_size, shuffle = False, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size, shuffle = False, num_workers=8)

def _train(epoch):
    train_loss = 0
    for x in tqdm(train_loader):
        x = x.to(device)
        loss = model.train({"x": x})
        train_loss += loss
        
    train_loss = train_loss
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
    
    return train_loss

def _val(epoch):
    val_loss = 0
    for x in val_loader:
        x = x.to(device)       
        loss = model.test({"x": x})
        val_loss += loss
        
        pred_y = f.sample_mean({"x": x}).argmax(1).detach().cpu().numpy()

    test_loss = val_loss
    print('Test loss: {:.4f}'.format(val_loss))
    return val_loss

def train(epochs):
    for epoch in range(1, epochs + 1):
        train_loss = _train(epoch)
        val_loss = _val(epoch)

train(100)


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def draw(image_path, out_path):
    with Image.open(image_path) as img:
        w, h = img.size
    dataset = TestDS2D(image_path, kernel_size)
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    pred_ys = []
    #u_coords = []
    #v_coords = []
    for x, ind in loader:
        x = x.to(device)
        pred_y = f.sample_mean({"x": x}).argmax(1).detach().cpu()
        pred_ys.append(pred_y)
        #u = ind[0]
        #v = ind[1]
        #u_coords.append(u)
        #v_coords.append(v)
    
    seg_image = torch.cat(pred_ys).reshape([w,h]).numpy()
    cmap = plt.get_cmap("jet", 30)
    plt.imshow(seg_image, cmap)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.cla()

import os
if not os.path.exists("result"):
    os.mkdir("result/2010")
    os.mkdir("result/2020")

for path in glob("crop/2010/*"):
    draw(path, path.replace("crop", "result"))


class GMVAE2D_US:
    """
    GMVAE object for unsupervised 2D (single) image segmentation.
    """
    def __init__(self, data_dir, kernel_size=(9,9), z_dim=64, batch_size=500, device = "cuda"):
        """
        Parameters
        ----------
        data_dir : str
            A data directory.
        kernel_size : tuple of int default (9, 9)
            A kernel size.
        z_dim : int default 64
            A latent variable dimension
        batch_size : int default 500
            A batch size
        device : str default "cuda"
            A device to use with pytorch.
        """
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.device = device

        # set a model
        x_dim = self.kernel_size[0] * self.kernel_size[1] * 3
        
        self.p = Generator2D(x_dim, z_dim).to(device)
        self._q = Inference2D(x_dim, z_dim, y_dim=num_cluster).to(device)
        self.f = Classifier2D(x_dim, num_cluster).to(device)
        self.q = self._q * self.f
        self.prior = Prior2D(z_dim, y_dim=num_cluster).to(device)
        self.p_joint = self.p * self.prior
            
        elbo = ELBO(p_joint, q)
        loss_cls = -elbo.mean()
        
        self.model = Model(loss_cls,test_loss=loss_cls,
                      distributions=[p, _q, f, prior], 
                      optimizer=optim.Adam, optimizer_params={"lr":1e-3})
        print(self.model)

        dataset = TrainDS2D(data_dir, kernel_size)
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
    
        test_loss = val_loss
        print('Test loss: {:.4f}'.format(val_loss))
        return val_loss
    
    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch)
            val_loss = self._val(epoch)
      
    def draw(self, image_path, out_path):
        with Image.open(image_path) as img:
            w, h = img.size
        dataset = TestDS2D(image_path, kernel_size)
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
        pred_ys = []
        #u_coords = []
        #v_coords = []
        for x, ind in loader:
            x = x.to(device)
            pred_y = f.sample_mean({"x": x}).argmax(1).detach().cpu()
            pred_ys.append(pred_y)
            #u = ind[0]
            #v = ind[1]
            #u_coords.append(u)
            #v_coords.append(v)
        
        seg_image = torch.cat(pred_ys).reshape([w,h]).numpy()
        cmap = plt.get_cmap("jet", 30)
        plt.imshow(seg_image, cmap)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.cla()
            
gmvae = GMVAE2D_US(data_dir="crop/2010", kernel_size=(3,3), z_dim=128, batch_size=50000, device="cuda")

gmvae.train(50)