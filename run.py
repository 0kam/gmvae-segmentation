from models import GMVAE2D_US, GMVAE3D_US, GMVAE2D_SS
from utils import save_ss_data2D
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

gmvae.draw("data/crop/2010", "result/3D_US/2010_kernel55_cluster50_ep1000.png")

# Semisupervised
save_ss_data2D("data/tateyama.json", "data/tateyama.jpeg", "data/semisup/tateyama/", (9,9), 2000)

gmvae = GMVAE2D_SS("data/semisup/tateyama/", z_dim=16, batch_size=5, device="cuda", num_workers=0)
gmvae.train(15)

seg = gmvae.draw("data/tateyama.jpeg", "tateyama_ss.jpg", (9, 9), batch_size = 50000, num_workers=20)