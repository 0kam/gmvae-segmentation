from torch.distributions import Dirichlet
import torch

m = Dirichlet(torch.tensor([[1.0,1.0],[1.0,1.0]]))
m.sample()