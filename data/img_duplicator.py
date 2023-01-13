import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import glob
import numpy as np

# uはUnicodeって意味らしい
CATEGORIES = [
    u'16Tea',
    u'AYATAKA',
    u'AYATAKA_amami',
    u'Nagomi',
    u'OiOcha_shinryoku',
    u'BarleyTea_Tsurube',
    u'OiOcha',]

# 画像サイズ
IMG_SIZE = 150

# ImageDataGeneratorを定義
# horizontalは画像の回転
# zoomは拡大の度合い
DATA_GENERATOR = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=0.3, zoom_range=0.1)

#画像をまとめているファイルまでのパス
IMG_ROOT_DIR = './images'

#idxにはインデックス番号、categoryには'16Tea'などが入る
for idx, category in enumerate(CATEGORIES):  
    # コピー元
    img_dir = os.path.join(IMG_ROOT_DIR, 'original', category)
    # コピー先
    out_dir = os.path.join(IMG_ROOT_DIR, 'extended', category)
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(img_dir, '*.jpeg'))
    for i, file in enumerate(files):
        img = keras.preprocessing.image.load_img(file)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        # imageをndarrayに変換
        x = keras.preprocessing.image.img_to_array(img)
        # xというndarrayの0の位置に次元を追加する
        x = np.expand_dims(x, axis=0)
        # flowでミニバッチを生成する
        g = DATA_GENERATOR.flow(x, batch_size=1, save_to_dir=out_dir, save_prefix='img', save_format='jpg')
        # 元の画像から少し(0.3) horizontal_flipして，少し(0.1) zoomした画像を10枚作る
        for i in range(10):
            batch = g.next()
    print(u'{} : ファイル数は {} 件です。'.format(category, len(os.listdir(out_dir))))
