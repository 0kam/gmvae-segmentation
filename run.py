from models import GMVAE2D_US, GMVAE3D_US
from glob import glob

gmvae = GMVAE2D_US(data_dir="crop/2010", kernel_size=(5,5), num_cluster = 10,
    z_dim=64, batch_size=5000, device="cuda")
gmvae.train(300)

import os
if not os.path.exists("result"):
    os.mkdir("result/2D_US/2010")
    os.mkdir("result/2D_US/2020")

for path in glob("data/crop/2010/*"):
    gmvae.draw(path, path.replace("data/crop", "result/2D_US"))

gmvae = GMVAE3D_US(data_dir="data/crop", kernel_size=(5,5), num_cluster = 50,
    z_dim=4, batch_size=5000, device="cuda", num_workers = 20)

x = iter(gmvae.train_loader).next()
gmvae.train(1000)

gmvae.draw("crop/2010", "2010_kernel55_cluster20_ep300.png")