# $ python main.py './test/test1.jpeg'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
import os
import cv2

# パスの整理
current_dir = os.getcwd()
MODEL_PATH = os.path.join(current_dir, 'data/models/model_predict.json')
WEIGHT_PATH = os.path.join(current_dir, 'data/models/model_predict.hdf5')
# 入力引数から画像を読み込む
args = sys.argv

# 顔領域の切り取り
#顔検出器のインポート
import dlib
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


'''
# 画像サイズ
IMG_SIZE = 150
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE,3)

# モデルを読み込む
model = keras.models.model_from_json(open(MODEL_PATH).read())
model.load_weights(WEIGHT_PATH)



# 読み込んだ画像をINPUT_SHAPEにリサイズ
img = keras.preprocessing.image.load_img(args[1], target_size=INPUT_SHAPE)
# 前の行でリサイズしたimgをndarrayに変換
x = keras.preprocessing.image.img_to_array(img)
# ndarrayに次元を追加（理由は不明）
x = np.expand_dims(x, axis=0)

# モデルで予測する
features = model.predict(x)
print(features)

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

'''