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

# 顔画像（カラー）の読み込み
face_imgs = glob.glob(os.path.join(face_cut_dir, '*.jpeg'))
# 切り抜きたい座標が入っているcsvを取得
df = pd.read_csv(os.path.join(current_dir, 'csv/lip_landmarks.csv'))

for f in face_imgs:

    img = np.array(Image.open(f))
    # 画像データをRGBからBGRへ変換
    img_bgr = img[:, :, ::-1]

    #　画像ファイル名と一致する座標を取り出す
    lip_landmark = df[df['file_name']==os.path.join(f)]['lip_landmarks']
    #　顔のランドマーク検出が上手くいってなくてcsvが空の時は処理しないようにする
    if( len(lip_landmark) > 0 ):
        # \nなどの余分なものを取り除く
        b = [list(map(int, s.split())) for s in [t for t in lip_landmark.str.replace("\\n", "").str.replace("]","").str.split("[") if t][0] if s]
        #　arrayに変換
        contour = np.array(b)

        # マスク画像を作成
        # 元の画像と同じ大きさのマスク画像を作る
        mask = np.zeros_like(img_bgr)
        cv2.fillConvexPoly(mask, contour, color=(255, 255, 255))

        # 背景画像
        bg_color = (0, 0, 0) # 黒
        bg_img = np.full_like(img_bgr, bg_color)

        # np.where() はマスクの値が (255, 255, 255) の要素は前景画像 img1 の値、
        # マスクの値が (0, 0, 0) の要素は背景画像 img2 の値を返す。
        result = np.where(mask==255, img_bgr, bg_img)
        file_name = f.replace('/images/face_cut/', '/images/lip_cut/')
        cv2.imwrite(file_name, result)
        