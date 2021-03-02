from torch import nn, optim
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pixyz.distributions import Normal, RelaxedCategorical
from pixyz.models import Model
from pixyz.losses import ELBO

# distributions for labeled data
## inference model q(z|x,y)

# distributions for labeled data
## inference model q(z|x,y)
class Inference2D(Normal):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__(cond_var=["x","y"], var=["z"], name="q")

        self.fc1 = nn.Linear(x_dim, 128)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(128 + y_dim, 128)
        self.prelu2 = nn.PReLU()
        self.fc31 = nn.Linear(128, z_dim)
        self.fc32 = nn.Linear(128, z_dim)

    def forward(self, x, y):
        h = self.prelu1(self.fc1(x))
        h = self.prelu2(self.fc2(torch.cat([h, y], 1)))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

# generative model p(x|z) 
class Generator2D(Normal):
    def __init__(self, x_dim, z_dim):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        self.fc1 = nn.Linear(z_dim, 128)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(128, 128)
        self.prelu2 = nn.PReLU()
        self.fc31 = nn.Linear(128, x_dim)
        self.fc32 = nn.Linear(128, x_dim)

    def forward(self, z):
        
        h = self.prelu1(self.fc1(z))
        h = self.prelu2(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}


# classifier p(y|x)
class Classifier2D(RelaxedCategorical):
    def __init__(self, x_dim, y_dim):
        super(Classifier2D, self).__init__(cond_var=["x"], var=["y"], name="p")
        
        self.fc1 = nn.Linear(x_dim, 128)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(128, 128)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(128, y_dim)

    def forward(self, x):
        h = self.prelu1(self.fc1(x))
        h = self.prelu2(self.fc2(h))
        h = F.softmax(self.fc3(h), dim=1)
        return {"probs": h}


# prior model p(z|y)
class Prior2D(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")

        self.fc1 = nn.Linear(y_dim, 32)
        self.prelu1 = nn.PReLU()
        self.fc21 = nn.Linear(32, z_dim)
        self.fc22 = nn.Linear(32, z_dim)
    
    def forward(self, y):
        h = self.prelu1(self.fc1(y))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

