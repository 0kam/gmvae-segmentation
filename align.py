from fpctools.align import homography_lensdist
import cv2
from glob import glob

destination = cv2.imread("2010/mrd_085_eos_vis_20101003_1200.jpg")

# 2010
files = glob("2010/*")
for f in files:
    img = cv2.imread(f)
    img, hmat, map_d, res = homography_lensdist(img, destination)
    cv2.imwrite(f, img)
