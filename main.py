# $ python main.py './test/test1.jpeg'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
import os
import cv2
import dlib
import glob
from imutils import face_utils
from PIL import Image

# パスの整理
current_dir = os.getcwd()
MODEL_PATH = os.path.join(current_dir, 'data/models/model_predict.json')
WEIGHT_PATH = os.path.join(current_dir, 'data/models/model_predict.hdf5')
# 入力引数から画像を読み込む
args = sys.argv

# 顔領域の切り取り
#顔検出器のインポート
faceDetector = dlib.get_frontal_face_detector()
#顔を切り抜く関数をdefで定義
def crop_face(path):
    img = cv2.imread(path)
    faces = faceDetector(img, 0)
    if len(faces) > 0:
        for i in range(0, len(faces)):# face_img には i 番目の顔のみが入る（他の顔の情報は無視される）
            face_img = img[int(faces[i].top()):int(faces[i].bottom()),int(faces[i].left()):int(faces[i].right())]
    return face_img

f = args[1]
try:
    face_img = crop_face(f)
    file_name = f.replace('test1', 'face_cut1')
    cv2.imwrite(file_name, face_img)
except Exception as e:
    print(f)
    print(e)


# 顔器官検出
# 顔検出器の学習済みモデルへのパス
predictor_path = os.path.join(current_dir, 'data/dlib/shape_predictor_68_face_landmarks.dat')
#画像のルートディレクトリ
img_root_dir = os.path.join(current_dir, 'data/images/')
#face_cutディレクトリへのパス
original_imgs_dir = os.path.join(img_root_dir, 'face_cut')
#ランドマーク検出した画像の保存先へのパス
result_imgs_dir = os.path.join(current_dir, 'test/')

# 顔検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()

# 顔のランドマーク検出ツールの呼び出し
face_predictor = dlib.shape_predictor(predictor_path)

# 検出対象の画像の呼び込み
img = cv2.imread('./test/face_cut1.jpeg')

#変数定義
landmarks = []

# 処理高速化のためグレースケール化(任意)
img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔検出
# ※2番めの引数はupsampleの回数。基本的に1回で十分。
faces = face_detector(img_gry, 1)

# （1枚の画像の中に複数の顔があった場合）検出した全顔に対して処理
for face in faces:
    # 顔のランドマーク検出
    landmark = face_predictor(img_gry, face)
    # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
    landmark = face_utils.shape_to_np(landmark)

    # 切り取りたい下唇の真ん中の座標を配列に詰める
    lip_landmarks = []
    lip_center = []
    #下唇の下側の座標（no.56~59）を逆にする（そうしないと変な形に切り抜かれるので）
    under_landmark = landmark[56:59]
    under_landmark = under_landmark[::-1]
    #リストにして詰める
    lip_landmarks = landmark[48:60].tolist()
    lip_center.extend(under_landmark.tolist())
    lip_center.extend(landmark[65:69].tolist())
    # ターミナルに出力
    # print(lip_landmarks)
    # print(lip_center)

# ランドマーク描画
for (i, (x, y)) in enumerate(lip_center):
    cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

# 生成した画像を保存
file_name = ('./test/lip_landmarks1.jpeg')
cv2.imwrite(file_name, img)


#唇領域を取り除く
f = './test/face_cut1.jpeg'
img = np.array(Image.open(f))
# 画像データをRGBからBGRへ変換
img_bgr = img[:, :, ::-1]

#　顔のランドマーク検出が上手くいってなくてcsvが空の時は処理しないようにする
try:
    # \nなどの余分なものを取り除く
    #b = [list(map(int, s.split())) for s in [t for t in lip_landmark.str.replace("\\n", "").str.replace("]","").str.split("[") if t][0] if s]
    #　arrayに変換
    # contour = np.array(b)
    # print(type(lip_landmark[0]), len(lip_landmark[0]), lip_landmark[0])
    contour = np.array(lip_landmarks)

    # マスク画像を作成
    # 元の画像と同じ大きさのマスク画像を作る
    bg_color = (255, 255, 255) # 黒
    mask = np.full_like(img_bgr, bg_color)
    cv2.fillConvexPoly(mask, contour, color=(0, 0, 0))
    
    # 背景画像
    bg_img = np.full_like(img_bgr, bg_color)

    # np.where() はマスクの値が (255, 255, 255) の要素は前景画像 img1 の値、
    # マスクの値が (0, 0, 0) の要素は背景画像の値を返す。
    result = np.where(mask==255, img_bgr, bg_img)
    file_name = './test/remove_lip1.jpeg'
    cv2.imwrite(file_name, result)
    
except Exception as e:
    print(type(e), os.path.basename(f))


# リップカラーの抽出

# 唇領域の切り取り

# マスク画像の作成
f = './test/face_cut1.jpeg'
img = np.array(Image.open(f))
# 画像データをRGBからBGRへ変換
img_bgr = img[:, :, ::-1]

try:
    # \nなどの余分なものを取り除く
    #b = [list(map(int, s.split())) for s in [t for t in lip_landmark.str.replace("\\n", "").str.replace("]","").str.split("[") if t][0] if s]
    #　arrayに変換
    # contour = np.array(b)
    # print(type(lip_landmark[0]), len(lip_landmark[0]), lip_landmark[0])
    contour = np.array(lip_center)

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
    file_name = './test/lip_cut1.jpeg'
    cv2.imwrite(file_name, result)
except Exception as e:
    print(type(e), os.path.basename(f))

# リップカラーの平均値の計算

bgr = []

img = cv2.imread('./test/lip_cut1.jpeg')
h, w, c = img.shape #height, width, channnel

# 初期値の定義
l=0
b_ave=0; g_ave=0; r_ave=0

for i in range(h):
    # 初期値の定義
    for j in range(w):
        #画素値[0,0,0]（Black）を除外してピクセルの和とbgrの画素値の合計を計算する
        #元はandじゃなくてor（色が完全に黒[0,0,0]のものだけ削除する）にしてた
        #けど、どうやら完全に黒じゃないっぽい（[0,0,1]とかが混じってる）のでandにした
        if(img[i,j,0] != 0 and img[i,j,1] != 0 and img[i,j,2] != 0 ):
            l+=1    #対象となるピクセル数を計算する
            #対象となるピクセルの画素値の和を計算する
            b_ave=b_ave+img[i,j,0]
            g_ave=g_ave+img[i,j,1]
            r_ave=r_ave+img[i,j,2]

#画素値合計をピクセル数で除することでRGBの画素値の平均値を求める
b_ave=b_ave/l
g_ave=g_ave/l
r_ave=r_ave/l

bgr.append([b_ave, g_ave, r_ave])
print('===lip_color_avg===')
print(bgr)

# 画像として保存
result_img_shape = (100, 100, 3)
lip_avg_img = np.full(result_img_shape, bgr)
file_name = './test/lip_avg.jpeg'
cv2.imwrite(file_name, lip_avg_img)


# テスト画像で推定

# 画像サイズ
IMG_SIZE = 150
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE,3)

# モデルを読み込む
model = keras.models.model_from_json(open(MODEL_PATH).read())
model.load_weights(WEIGHT_PATH)



# 読み込んだ画像をINPUT_SHAPEにリサイズ
img = keras.preprocessing.image.load_img('./test/remove_lip1.jpeg', target_size=INPUT_SHAPE)
# 前の行でリサイズしたimgをndarrayに変換
x = keras.preprocessing.image.img_to_array(img)
# ndarrayに次元を追加（理由は不明）
x = np.expand_dims(x, axis=0)

# モデルで予測する
features = model.predict(x)
print('===predict color===')
print(features)

# 画像として保存
predict_lip_img = np.full(result_img_shape, features)
file_name = './test/predict_lip.jpeg'
cv2.imwrite(file_name, predict_lip_img)

# コサイン類似度を計算する
bgr = np.array(bgr)
a = bgr.reshape((1,3))
features = np.array(features)
b = features.reshape((1,3))
cos_similarity = np.dot(a, b.T)/(np.sqrt(np.dot(a, a.T))*np.sqrt(np.dot(b, b.T)))
print('===cos_similarity===')
print(cos_similarity)


# print("確率：")
# for i in range(0, 7):
#     print(str(CATEGORIES_NAME[i]) + ' ： ' + str(features[0][i]))
# print("----------------------------------------------")
# print("計算結果")
# if np.argmax(features[0]) == 1:
#     print(u'選ばれたのは綾鷹でした。')
# else:
#     print(u'綾鷹ではなく' + str(CATEGORIES_NAME[np.argmax(features[0])]) + 'が選ばれました。')
# print("----------------------------------------------")
