import cv2
import numpy as np
import os
import uuid
import glob

face_cut_imgs_dir = os.path.join(os.getcwd(),"images/face_cut")
face_color_imgs_dir = os.path.join(os.getcwd(),"images/face_color/")
os.makedirs(face_color_imgs_dir, exist_ok=True)

#face_cutディレクトリ内の画像を全て読み込む
files = glob.glob(os.path.join(face_cut_imgs_dir, '*.jpeg'))

# define parameter
HSV_MIN = np.array([0, 30, 70])
HSV_MAX = np.array([20, 150, 255])
# ソースコードの色番
# HSV_MIN = np.array([0, 30, 60])
# HSV_MAX = np.array([20, 150, 255])

for f in files:

    img = cv2.imread(f)

    #convert hsv
    #rgb画像をhsv画像に変更
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #mask hsv region
    #肌色領域をマスク
    mask_hsv = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX)

    # save image
    #print(face_color_imgs_dir + str(uuid.uuid4()) + str('.jpeg'))
    cv2.imwrite(face_color_imgs_dir + str(uuid.uuid4()) + str('.jpeg'), mask_hsv)
