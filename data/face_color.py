import cv2
import numpy as np
import os
import uuid
import glob

face_cut_imgs_dir_path = os.path.join(os.getcwd(),"data/face_cut_imgs/")
face_color_imgs_dir_path = os.path.join(os.getcwd(),"data/face_color_imgs/")

#data/face_cut_imgs内の画像を全て読み込む
files = glob.glob(str(face_cut_imgs_dir_path) + str("*"))

# define parameter
HSV_MIN = np.array([0, 30, 70])
HSV_MAX = np.array([20, 150, 255])
# ソースコードの色番
# HSV_MIN = np.array([0, 30, 60])
# HSV_MAX = np.array([20, 150, 255])

for f in files:

    img = cv2.imread(f)

    #convert hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #mask hsv region
    mask_hsv = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX)

    # save image
    # cv2.imwrite("mask_hsv.jpg", mask_hsv)
    cv2.imwrite(face_color_imgs_dir_path + str(uuid.uuid4()) + str('.jpeg'), mask_hsv)
