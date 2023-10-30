# python main.py './data/images/test/ファイル名'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import sys
import os

# パスの整理
current_dir = os.getcwd()
MODEL_PATH = os.path.join(current_dir, '/data/models/model_predict.json')
WEIGHT_PATH = os.path.join(current_dir, '/data/models/model_predict.hdf5')

# カテゴリ
CATEGORIES = [
    u'16Tea',
    u'AYATAKA',
    u'AYATAKA_amami',
    u'Nagomi',
    u'OiOcha_shinryoku',
    u'BarleyTea_Tsurube',
    u'OiOcha',]

CATEGORIES_NAME = [
    u'十六茶',
    u'綾鷹',
    u'綾鷹茶葉のあまみ',
    u'おいしい緑茶なごみ',
    u'おーいお茶新緑',
    u'天然ミネラル麦茶',
    u'おーいお茶']

# 画像サイズ
IMG_SIZE = 150
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE,3)

# モデルを読み込む
model = keras.models.model_from_json(open(MODEL_PATH).read())
model.load_weights(WEIGHT_PATH)

# 入力引数から画像を読み込む
args = sys.argv
img = keras.preprocessing.image.load_img(args[1], target_size=INPUT_SHAPE)
x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# モデルで予測する
features = model.predict(x)
print("確率：")
for i in range(0, 7):
    print(str(CATEGORIES_NAME[i]) + ' ： ' + str(features[0][i]))
print("----------------------------------------------")
print("計算結果")
if np.argmax(features[0]) == 1:
    print(u'選ばれたのは綾鷹でした。')
else:
    print(u'綾鷹ではなく' + str(CATEGORIES_NAME[np.argmax(features[0])]) + 'が選ばれました。')
print("----------------------------------------------")
