import numpy as np
import cv2
import os
import glob
import pandas as pd

# パスの定義
# 画像のルートディレクトリ
current_dir = os.getcwd()
img_root_dir = os.path.join(current_dir, 'images/')
# 顔だけを切り抜いた画像のパス
face_cut_dir = os.path.join(img_root_dir, 'face_cut')
# マスク後の画像の保存場所
masked_imgs_dir = os.path.join(img_root_dir, 'face_masked')
os.makedirs(masked_imgs_dir, exist_ok=True)

# 顔画像（カラー）の読み込み
color_files = glob.glob(os.path.join(face_cut_dir, '*.jpeg'))

# 画像をマスクして肌色領域のみの画像を生成
for f in color_files:

    color_img = cv2.imread(f)

    try:
        # 処理している画像と同じ名前のマスク画像を読み込む
        mask_img_path = f.replace('/images/face_cut/', '/images/face_color/')
        mask_img = cv2.imread(mask_img_path)
        # そのままだと画像サイズが違ってエラーが出るので、グレースケールに変換
        #mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        # 画像をマスクして保存
        img_AND = cv2.bitwise_and(color_img, mask_img)
        file_name = f.replace('/images/face_cut/', '/images/face_masked/')
        cv2.imwrite(file_name, img_AND)

    except Exception as e:
        print(f)
        print(e)

# 肌色領域の色の平均値の計算

# マスク済み画像の読み込み
masked_imgs = glob.glob(os.path.join(masked_imgs_dir, '*.jpeg'))

#変数定義
input_dir = masked_imgs_dir
output_dir = current_dir
num_photo = sum(os.path.isfile(os.path.join(masked_imgs_dir, name)) for name in os.listdir(masked_imgs_dir))
bgr = np.zeros((num_photo,4))
file_number = 0 #csvを出力する時の通し番号

for k in masked_imgs:
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

    bgr[file_number]=np.array([file_number, b_ave, g_ave, r_ave])
    file_number = file_number + 1

df = pd.DataFrame(bgr, columns=['file_number', 'blue', 'green', 'red'])    #opencvの並び準BGRに合わせる
df.to_csv(output_dir + '/face_color_avg.csv')


