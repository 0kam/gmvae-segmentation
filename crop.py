from PIL import Image
from glob import glob

# 2010
files = glob("three/2010/*")
for f in files:
    im = Image.open(f)
    img = im.crop((2150, 1800, 2650, 2300))
    f_crop = f.replace("/2010/", "/crop/2010/")
    img.save(f_crop)

# 2020
files = glob("three/2020/*")
for f in files:
    im = Image.open(f)
    img = im.crop((2150, 1800, 2650, 2300))
    f_crop = f.replace("/2020/", "/crop/2020/")
    img.save(f_crop)
