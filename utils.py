from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

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
            self.target_image = Image.open(self.target_image_path) 
            # print("Read a new image " + self.image_paths[idx // (self.size[0] * self.size[1])])
        center = (pix_num % self.size[1], pix_num // self.size[0])
        kw = int((self.kernel_size[0]-1) / 2)
        kh = int((self.kernel_size[1]-1) / 2)
        left = max(center[0] - kw, 0)
        upper = max(center[1] - kh, 0)
        right = min(center[0] + kw, self.size[0])
        lower = min(center[1] + kh, self.size[1])
        patch = to_tensor(self.target_image.crop((left, upper, right, lower)).resize(self.kernel_size)).view(-1)
        
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
        self.target_image = Image.open(self.image_path)
        self.data_length = self.target_image.width * self.target_image.height
        self.size = self.target_image.size

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
        patch = to_tensor(self.target_image.crop((left, upper, right, lower)).resize(self.kernel_size)).view(-1)
        
        return patch, center
