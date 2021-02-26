import cv2
from glob import glob
import numpy as np

files2010 = glob("2010/*")
files2020 = glob("2020/*")

for i in range(len(files2010)):
    f_2010 = files2010[i]
    f_2020 = files2020[i]
    img_2010 = cv2.imread(f_2010)
    img_2020 = cv2.imread(f_2020)

    img_2010 = cv2.cvtColor(img_2010, cv2.COLOR_BGR2HSV)
    img_2020 = cv2.cvtColor(img_2020, cv2.COLOR_BGR2HSV)

    diff = np.mean(img_2010[:,:,2]) - np.mean(img_2020[:,:,2])
    img_2020[:,:,2] = img_2020[:,:,2] + diff
    img_2020 = cv2.cvtColor(img_2020, cv2.COLOR_HSV2BGR)
    f_res = f_2020.replace(".jpg", "_norm.jpg")
    cv2.imwrite(f_res, img_2020)

