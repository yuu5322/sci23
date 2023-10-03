import cv2
import numpy as np

# img1: 元の画像, 形状が (Width, Height, 3) の numpy 配列
# contour: 輪郭の頂点情報, 形状が (NumPoints, 2) の numpy 配列

# マスク画像を作成する。
# 前景の画素は (255, 255, 255)、背景の画素は (0, 0, 0)
mask = np.zeros_like(img1)
cv2.fillConvexPoly(mask, contour, color=(255, 255, 255))

# 背景画像
bg_color = (50, 200, 0)
img2 = np.full_like(img, bg_color)

# np.where() はマスクの値が (255, 255, 255) の要素は前景画像 img1 の値、
# マスクの値が (0, 0, 0) の要素は背景画像 img2 の値を返す。
result = np.where(mask==255, img1, img2)