import cv2
from PIL import Image
import numpy as np
import os
import glob
import pandas as pd
import json

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
    lip_landmark = df[df['file_name']==os.path.join(f)]['lip_landmarks'].tolist()
    #　顔のランドマーク検出が上手くいってなくてcsvが空の時は処理しないようにする
    try:
        # \nなどの余分なものを取り除く
        #b = [list(map(int, s.split())) for s in [t for t in lip_landmark.str.replace("\\n", "").str.replace("]","").str.split("[") if t][0] if s]
        #　arrayに変換
        # contour = np.array(b)
        # print(type(lip_landmark[0]), len(lip_landmark[0]), lip_landmark[0])
        contour = np.array(json.loads(lip_landmark[0]))

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
    except Exception as e:
        print(type(e), os.path.basename(f))


# 唇領域の色の平均値の計算

# マスク済み画像の読み込み
lip_imgs = glob.glob(os.path.join(lip_cut_dir, '*.jpeg'))

#変数定義
input_dir = lip_cut_dir
output_dir = os.path.join(current_dir, 'csv/')
num_photo = sum(os.path.isfile(os.path.join(lip_cut_dir, name)) for name in os.listdir(lip_cut_dir))
#numpy配列で定義すると1種類のデータ型しか追加できなくなるので、空の配列で定義する
bgr = []
#csvを出力する時の通し番号
file_number = 0

for k in lip_imgs:
    img = cv2.imread(k)
    h, w, c = img.shape #height, width, channnel

    #初期化
    l=0
    b_ave=0; g_ave=0; r_ave=0

    for i in range(h):
        for j in range(w):
            #画素値[0,0,0]（Black）を除外してピクセルの和とbgrの画素値の合計を計算する
            if(img[i,j,0] != 0 or img[i,j,1] != 0 or img[i,j,2] != 0 ):
                l+=1    #対象となるピクセル数を計算する
                #対象となるピクセルの画素値の和を計算する
                b_ave=b_ave+img[i,j,0]
                g_ave=g_ave+img[i,j,1]
                r_ave=r_ave+img[i,j,2]

    #画素値合計をピクセル数で除することでRGBの画素値の平均値を求める
    b_ave=b_ave/l
    g_ave=g_ave/l
    r_ave=r_ave/l

    #np配列だった場合はbgr[file_number]=np.array([file_number, b_ave, g_ave, r_ave])
    file_name_for_csv = k.replace('/images/lip_cut/', '/images/face_cut/')
    bgr.append([file_number, file_name_for_csv, b_ave, g_ave, r_ave])
    file_number = file_number + 1

df = pd.DataFrame(bgr, columns=['file_number', 'filename', 'blue', 'green', 'red'])    #opencvの並び準BGRに合わせる
df.to_csv(output_dir + '/lip_color_avg.csv')
