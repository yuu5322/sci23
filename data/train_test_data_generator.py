# -*- coding: utf-8 -*-

from PIL import Image
import os, glob
import numpy as np
import random, math
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 画像が保存されているルートディレクトリのパス
IMG_ROOT_DIR = './images/extended'
# カテゴリ
CATEGORIES = [
    u'16Tea',
    u'AYATAKA',
    u'AYATAKA_amami',
    u'Nagomi',
    u'OiOcha_shinryoku',
    u'BarleyTea_Tsurube',
    u'OiOcha',]

# 密度
DENSE_SIZE = len(CATEGORIES)
# 画像サイズ
IMG_SIZE = 150
# 画像データ
X = []
# カテゴリデータ
Y = []
# 教師データ
X_TRAIN = []
Y_TRAIN = []
# テストデータ
X_TEST = []
Y_TEST = []
# 生成するファイルの名前
TRAIN_TEST_DATA = './images/train_test_data/data.npy'


# カテゴリごとに処理する
# enumerateはリストのインデックス番号と要素の両方を取得できるようにするやつ
for idx, category in enumerate(CATEGORIES):
    # 各ラベルの画像ディレクトリにアクセス
    image_dir = os.path.join(IMG_ROOT_DIR, category)
    #.jpgで終わるファイルを全部取得してfilesの中に入れる
    files = glob.glob(os.path.join(image_dir, '*.jpg'))
    #filesでforループ
    for f in files:
        # 各画像をリサイズしてデータに変換する
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        data = np.asarray(img)
        # appendはリストに要素を追加するやつ
        X.append(data)
        Y.append(idx)

X = np.array(X)
Y = np.array(Y)

# 正規化
X = X.astype('float32') /255
#one-hotベクトルにする、DENSE_SIZEが最大値
Y = keras.utils.to_categorical(Y, DENSE_SIZE)

#XとYをプリントデバッグ
print(X)
print(Y)

# 教師データとテストデータを分ける
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.20)

# 教師／テストデータを保存する
np.save(TRAIN_TEST_DATA, (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST))
print(u'教師／テストデータの作成が完了しました。: {}'.format(TRAIN_TEST_DATA))
