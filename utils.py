from glob import glob
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import pandas as pd
import json
import cv2
import numpy as np

class TrainDS2D(Dataset):
    """
    Train Dataset that returns patches of images with a given kernel_size.
    Attributes
    ----------
    data_dir : str
        An input data directory. All images must be same size.
    kernel_size : tuple of int
        A tuple (width, height) of the kernel.
    """
    def __init__(self, data_dir, kernel_size):
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.image_paths = glob(data_dir + "/*")
        data_length = 0
        for f in self.image_paths:
            with Image.open(f) as img:
                w, h = img.size
                data_length += w * h
        self.data_length = data_length
        with Image.open(f) as img:
            self.size = img.size
        self.target_image = None
        self.target_image_path = None

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        pix_num = idx % (self.size[0] * self.size[1])
        to_read = self.image_paths[idx // (self.size[0] * self.size[1])]
        if self.target_image == None or to_read != self.target_image_path:
            self.target_image_path = to_read
            self.target_image = to_tensor(Image.open(self.target_image_path)) 
            # print("Read a new image " + self.image_paths[idx // (self.size[0] * self.size[1])])
        center = (pix_num % self.size[1], pix_num // self.size[0])
        kw = int((self.kernel_size[0]-1) / 2)
        kh = int((self.kernel_size[1]-1) / 2)
        left = max(center[0] - kw, 0)
        upper = max(center[1] - kh, 0)
        right = min(center[0] + kw, self.size[0])
        lower = min(center[1] + kh, self.size[1])
        #patch = to_tensor(self.target_image.crop((left, upper, right, lower)).resize(self.kernel_size)).view(-1)
        patch = self.target_image[:,upper:lower,left:right]
        patch = F.interpolate(patch.unsqueeze(0), [self.kernel_size[1], self.kernel_size[0]])
        patch = patch.view(-1)
        return patch


class TestDS2D(Dataset):
    """
    Test Dataset that returns patch, width and height of the patch center.
    This can be applied to a single image.
    
    Attributes
    ----------
    image_path : str
        An input image path.
    kernel_size : tuple of int
        A tuple (width, height) of the kernel.
    """
    def __init__(self, image_path, kernel_size):
        self.kernel_size = kernel_size
        self.image_path = image_path
        timg = Image.open(self.image_path)
        self.target_image = to_tensor(timg)
        self.data_length = timg.width * timg.height
        self.size = timg.size

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        pix_num = idx % (self.size[0] * self.size[1])
        center = (pix_num % self.size[1], pix_num // self.size[0])
        kw = int((self.kernel_size[0]-1) / 2)
        kh = int((self.kernel_size[1]-1) / 2)
        left = max(center[0] - kw, 0)
        upper = max(center[1] - kh, 0)
        right = min(center[0] + kw, self.size[0])
        lower = min(center[1] + kh, self.size[1])
        #patch = to_tensor(self.target_image.crop((left, upper, right, lower)).resize(self.kernel_size)).view(-1)
        patch = self.target_image[:,upper:lower,left:right]
        patch = F.interpolate(patch.unsqueeze(0), [self.kernel_size[1], self.kernel_size[0]])
        patch = patch.view(-1)

        return patch, center


class TrainDS3D(Dataset):
    """
    Train Dataset that returns time series patches of images with a given kernel_size.
    Attributes
    ----------
    data_dir : str
        An input data directory that has subdrirectories with time-series images. All images must be same size.
    kernel_size : tuple of int
        A tuple (width, height) of the kernel.
    """
    def __init__(self, data_dir, kernel_size):
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.image_dirs = glob(data_dir + "/*")
        f = glob(self.image_dirs[0] + "/*")[0]
        with Image.open(f) as img:
            self.size = img.size
            self.data_length = img.width*img.height*len(self.image_dirs)
        self.target_images = None
        self.target_dir = None

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        pix_num = idx % (self.size[0] * self.size[1])
        to_read = self.image_dirs[idx // (self.size[0] * self.size[1])]
        if self.target_images == None or to_read != self.target_dir:
            self.target_dir = to_read
            self.target_images = torch.stack([to_tensor(Image.open(f)) for f in glob(self.target_dir + "/*")], dim = 0)
            # print("Read new images " + self.target_dir)
        center = (pix_num % self.size[1], pix_num // self.size[0])
        kw = int((self.kernel_size[0]-1) / 2)
        kh = int((self.kernel_size[1]-1) / 2)
        left = max(center[0] - kw, 0)
        upper = max(center[1] - kh, 0)
        right = min(center[0] + kw, self.size[0])
        lower = min(center[1] + kh, self.size[1])
        patch = self.target_images[:,:,upper:lower,left:right]
        patch = F.interpolate(patch, [self.kernel_size[1], self.kernel_size[0]])
        patch = torch.reshape(patch, (patch.shape[0], -1))
        return patch


class TestDS3D(Dataset):
    """
    Train Dataset that returns time series patches of images with a given kernel_size.
    Attributes
    ----------
    image_dir : str
        An input data directory that has a set of time-series images. All images must be same size.
    kernel_size : tuple of int
        A tuple (width, height) of the kernel.
    """
    def __init__(self, image_dir, kernel_size):
        self.image_dir = image_dir
        self.kernel_size = kernel_size
        f = glob(self.image_dir + "/*")[0]
        with Image.open(f) as img:
            self.size = img.size
            self.data_length = img.width*img.height
        self.target_images = torch.stack([to_tensor(Image.open(f)) for f in glob(self.image_dir + "/*")], dim = 0)
    
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        pix_num = idx % (self.size[0] * self.size[1])

        center = (pix_num % self.size[1], pix_num // self.size[0])
        kw = int((self.kernel_size[0]-1) / 2)
        kh = int((self.kernel_size[1]-1) / 2)
        left = max(center[0] - kw, 0)
        upper = max(center[1] - kh, 0)
        right = min(center[0] + kw, self.size[0])
        lower = min(center[1] + kh, self.size[1])
        patch = self.target_images[:,:,upper:lower,left:right]
        patch = F.interpolate(patch, [self.kernel_size[1], self.kernel_size[0]])
        patch = torch.reshape(patch, (patch.shape[0], -1))
        return patch

def read_sse(json_path, img_path):
    with open(json_path) as js:
        obj = json.load(js)["objects"]
    obj = pd.io.json.json_normalize(obj)
    polygons = list(obj["polygon"])
    labels = obj["classIndex"]
    img = cv2.imread(img_path)
    img[:,:,:] = 0
    for p, l in zip(polygons, labels):
        polygon = []
        for point in p:
            point = list(point.values())
            polygon.append(point)
        polygon = np.array(polygon).reshape(-1, 1, 2).astype("int32")
        img = cv2.fillConvexPoly(img, polygon, color = (l, l, l))


    

