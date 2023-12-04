from PIL import Image
import sys, os, glob, csv
import numpy as np
import random, math
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


# パスの定義
# リップカラーのcsvデータ
current_dir = os.getcwd()
lip_color_csv = os.path.join(current_dir, 'csv/lip_color_avg.csv')
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

LIP_COLOR_DATA_FILE = lip_color_csv
lip_color_data = []
# csvファイルを読み込みモード（"r"）で開く
# fは任意の名前（fじゃなくてもいい）
with open(LIP_COLOR_DATA_FILE, "r") as f:
    header = True
    reader = csv.reader(f)
    for row in reader:
        if header:
            header = False
            continue
        lip_color_data.append({"id":int(row[0]), "file":row[2], "blue":float(row[3]), "green":float(row[4]), "red":float(row[5])})

# 密度（Dense Layerの出力ニューロン数）
# 今回はB, G, Rの3つの数値を出力するので3
DENSE_SIZE = 3
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


#lip_color_dataでforループ
# entryは要素のかたまりという意味（任意の名前）
for entry in lip_color_data:
    # face_cutフォルダから取得した画像をリサイズしてデータに変換する
    img = Image.open(entry["file"])
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    data = np.asarray(img)
    # appendはリストに要素を追加するやつ
    # Xはdata、つまりface_cutフォルダ内の画像をnparrayに変換したもの
    # Yはbgr、つまりリップカラーの数値
    X.append(data)
    Y.append([entry["blue"], entry["green"], entry["red"]])

X = np.array(X)
Y = np.array(Y, dtype=np.float32)

# 正規化
# 綾鷹はベクトルにしてたけど今回はしない
X = X.astype('float32') / 255
Y = Y / 255

#XとYをプリントデバッグ
print(X)
print(Y)

# 教師データとテストデータを分ける
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.20)

# 教師／テストデータを保存する
np.save(TRAIN_TEST_DATA, (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST))
print(u'教師／テストデータの作成が完了しました。: {}'.format(TRAIN_TEST_DATA))
