import cv2
from PIL import Image
import numpy as np
import os
import glob
import pandas as pd

# パスの定義
# 画像のルートディレクトリ
current_dir = os.getcwd()
img_root_dir = os.path.join(current_dir, 'images/')
# 顔だけを切り抜いた画像のパス
face_cut_dir = os.path.join(img_root_dir, 'face_cut')
# 唇領域のみの画像の保存場所
lip_cut_dir = os.path.join(img_root_dir, 'lip_cut')
os.makedirs(lip_cut_dir, exist_ok=True)

# img1: 元の画像（とりあえず画像を1枚指定）
img1 = np.array(Image.open(os.path.join(face_cut_dir, '0a6d8bbe-bff4-4e44-afe1-730d7161992c.jpeg')))
# 画像データをRGBからBGRへ変換
img1 = img1[:, :, ::-1]
# contour: 切り抜きたい形の輪郭の頂点情報, 形状が (NumPoints, 2) の numpy 配列
contour = np.array((( 67, 148),
( 73, 140),
( 83, 134),
( 92, 135),
(101, 131),
(117, 132),
(134, 134),
(123, 150),
(110, 159),
(100, 161),
( 90, 162)))

# 座標を取得
lip_landmark = pd.read_csv(os.path.join(current_dir, 'csv/lip_landmarks.csv')).head(3)
print(lip_landmark)

# マスク画像を作成
# 元の画像と同じ大きさのマスク画像を作る
mask = np.zeros_like(img1)
cv2.fillConvexPoly(mask, contour, color=(255, 255, 255))

# 背景画像
bg_color = (0, 0, 0) # 黒
img2 = np.full_like(img1, bg_color)

# np.where() はマスクの値が (255, 255, 255) の要素は前景画像 img1 の値、
# マスクの値が (0, 0, 0) の要素は背景画像 img2 の値を返す。
result = np.where(mask==255, img1, img2)
file_name = '0a6d8bbe-bff4-4e44-afe1-730d7161992c.jpeg'
cv2.imwrite(os.path.join(lip_cut_dir, file_name), result)