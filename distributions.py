from torch import nn, optim
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pixyz.distributions import Normal, RelaxedCategorical, Deterministic
from pixyz.models import Model
from pixyz.losses import ELBO

# Distributions for 2D (single) images.
## inference model q(z|x,y)
class Inference2D(Normal):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__(cond_var=["x","y"], var=["z"], name="q")

        h_dim = int((x_dim + z_dim) / 2)
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(h_dim + y_dim, h_dim)
        self.prelu2 = nn.PReLU()
        self.fc31 = nn.Linear(h_dim, z_dim)
        self.fc32 = nn.Linear(h_dim, z_dim)

    def forward(self, x, y):
        h = self.prelu1(self.fc1(x))
        h = self.prelu2(self.fc2(torch.cat([h, y], 1)))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

# generative model p(x|z) 
class Generator2D(Normal):
    def __init__(self, x_dim, z_dim):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        h_dim = int((x_dim + z_dim) / 2)
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.prelu2 = nn.PReLU()
        self.fc31 = nn.Linear(h_dim, x_dim)
        self.fc32 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        
        h = self.prelu1(self.fc1(z))
        h = self.prelu2(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}


# classifier p(y|x)
class Classifier2D(RelaxedCategorical):
    def __init__(self, x_dim, y_dim):
        super(Classifier2D, self).__init__(cond_var=["x"], var=["y"], name="p")
        
        h_dim = int((x_dim + y_dim) / 2)
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        h = self.prelu1(self.fc1(x))
        h = self.prelu2(self.fc2(h))
        h = F.softmax(self.fc3(h), dim=1)
        return {"probs": h}


# prior model p(z|y)
class Prior2D(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")

        h_dim = int((y_dim + z_dim) / 2)
        self.fc1 = nn.Linear(y_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
    
    def forward(self, y):
        h = self.prelu1(self.fc1(y))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


# Distributions for 3D (time-lapse) images.
## inference model q(z|x,y)
class Inference3D(Normal):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__(cond_var=["x","y"], var=["z"], name="q")

        self.h_dim = int((x_dim + z_dim) / 2)
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=self.h_dim, batch_first=True)
        self.fc1 = nn.Linear(self.h_dim, self.h_dim)
        self.prelu1 = nn.PReLU()
        self.fc21 = nn.Linear(self.h_dim, z_dim)
        self.fc22 = nn.Linear(self.h_dim, z_dim)

    def forward(self, x, y):
        _, h = self.lstm(x)
        h = h[0].view(-1, self.h_dim)
        h = self.prelu1(self.fc1(h))
        seq_length = torch.zeros(x.shape[1])
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

# generative model p(x|z) 
class Generator3D(Normal):
    def __init__(self, x_dim, z_dim, seq_length, device):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        h_dim = int((x_dim + z_dim) / 2)
        self.device = device
        self.seq_length = seq_length
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.lstm = nn.LSTM(h_dim, h_dim, batch_first=True)
        self.fc21 = nn.Linear(h_dim, x_dim)
        self.fc22 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        h = self.prelu1(self.fc1(z))
        h = h.unsqueeze(1)
        x1 = torch.zeros(h.shape).to(self.device)
        cell_state = torch.zeros(h.shape).transpose(0,1).to(self.device)
        init_hidden = h.transpose(0,1)
        x = []
        for _ in range(self.seq_length):
            _, hc = self.lstm(x1, (init_hidden, cell_state))
            x1 = hc[0].transpose(0,1)
            x.append(x1)
        x = torch.stack(x, dim=1).squeeze()

        return {"loc": self.fc21(x), "scale": F.softplus(self.fc22(x))}

# classifier p(y|x)
class Classifier3D(RelaxedCategorical):
    def __init__(self, x_dim, y_dim):
        super(Classifier3D, self).__init__(cond_var=["x"], var=["y"], name="p")
        
        self. h_dim = int((x_dim + y_dim) / 2)
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=self.h_dim, batch_first=True)
        self.fc1 = nn.Linear(self.h_dim, self.h_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(self.h_dim, y_dim)

    def forward(self, x):
        _, h = self.lstm(x)
        h = h[0].view(-1, self.h_dim)
        h = self.prelu1(self.fc1(h))
        h = F.softmax(self.fc2(h), dim=1)
        return {"probs": h}


# prior model p(z|y)
class Prior3D(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")

        h_dim = int((y_dim + z_dim) / 2)
        self.fc1 = nn.Linear(y_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
    
    def forward(self, y):
        h = self.prelu1(self.fc1(y))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


# Distributions for 3D (time-lapse) images.
## inference model q(z|x,y)
class Inference3D_dev(Normal):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__(cond_var=["x","y"], var=["z"], name="q")

        self.h_dim = int((x_dim + z_dim) / 2)
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=self.h_dim, batch_first=True)
        self.fc1 = nn.Linear(self.h_dim, self.h_dim)
        self.prelu1 = nn.PReLU()
        self.fc21 = nn.Linear(self.h_dim, z_dim)
        self.fc22 = nn.Linear(self.h_dim, z_dim)

    def forward(self, x, y):
        seq_length = x.shape[1]
        _, h = self.lstm(x)
        h = h[0].view(-1, self.h_dim)
        h = self.prelu1(self.fc1(h))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        
        loc = loc.unsqueeze(1)
        loc = loc.expand(-1,seq_length,-1)
        scale = scale.unsqueeze(1)
        scale = scale.expand(-1,seq_length,-1)
        return {"loc": loc, "scale": scale}

# generative model p(x|z) 
class Generator3D_dev(Normal):
    def __init__(self, x_dim, z_dim, device):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        h_dim = int((x_dim + z_dim) / 2)
        self.device = device
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.lstm = nn.LSTM(h_dim, h_dim, batch_first=True)
        self.fc21 = nn.Linear(h_dim, x_dim)
        self.fc22 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        seq_length = z.shape[1]
        z = z[:,0,:]
        h = self.prelu1(self.fc1(z))
        h = h.unsqueeze(1)
        x1 = torch.zeros(h.shape).to(self.device)
        cell_state = torch.zeros(h.shape).transpose(0,1).to(self.device)
        init_hidden = h.transpose(0,1)
        x = []
        for _ in range(seq_length):
            _, hc = self.lstm(x1, (init_hidden, cell_state))
            x1 = hc[0].transpose(0,1)
            x.append(x1)
        x = torch.stack(x, dim=1).squeeze()

        return {"loc": self.fc21(x), "scale": F.softplus(self.fc22(x))}

# classifier p(y|x)
class Classifier3D_dev(RelaxedCategorical):
    def __init__(self, x_dim, y_dim):
        super(Classifier3D, self).__init__(cond_var=["x"], var=["y"], name="p")
        
        self. h_dim = int((x_dim + y_dim) / 2)
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=self.h_dim, batch_first=True)
        self.fc1 = nn.Linear(self.h_dim, self.h_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(self.h_dim, y_dim)

    def forward(self, x):
        _, h = self.lstm(x)
        h = h[0].view(-1, self.h_dim)
        h = self.prelu1(self.fc1(h))
        h = F.softmax(self.fc2(h), dim=1)
        h = h.unsqueeze(1)
        h = h + 1e-7
        return {"probs": h}


# prior model p(z|y)
class Prior3D_dev(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")

        h_dim = int((y_dim + z_dim) / 2)
        self.fc1 = nn.Linear(y_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
    
    def forward(self, y):
        h = self.prelu1(self.fc1(y))
        h = h.unsqueeze(1)
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}