import cv2
from PIL import Image
import numpy as np

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
img1 = np.array(Image.open(os.path.join(face_cut_dir_root_dir, '0a6d8bbe-bff4-4e44-afe1-730d7161992c.jpeg')))
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