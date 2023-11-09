from PIL import Image
import os, glob
import numpy as np
import random, math
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# パスの定義
# 顔画像のディレクトリ
current_dir = os.getcwd()
face_cut_dir = os.path.join(current_dir, 'images/face_cut')
# テストデータ保存先
# ディレクトリ
train_test_data_dir = os.path.join(current_dir, 'train_test_data/')
os.makedirs(train_test_data_dir, exist_ok=True)# train_test_dataフォルダが存在しなかったら作成
# .npyファイル
TRAIN_TEST_DATA = os.path.join(train_test_data_dir, 'data.npy')
# data.npyファイルが存在しなかったら作成
file_path_obj = Path(TRAIN_TEST_DATA)
if not file_path_obj.exists():
    file_path_obj.touch()

# 密度（Dense Layerの出力ニューロン数）
# 綾鷹AIを引き継いでとりあえず10にしておく（後から調整可）
DENSE_SIZE = 10
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


#.jpgで終わるファイルを全部取得してfilesの中に入れる
files = glob.glob(os.path.join(face_cut_dir, '*.jpeg'))
#filesでforループ
for f in files:
    # 各画像をリサイズしてデータに変換する
    img = Image.open(f)
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    data = np.asarray(img)
    # appendはリストに要素を追加するやつ
    X.append(data)
    Y.append(1)

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
