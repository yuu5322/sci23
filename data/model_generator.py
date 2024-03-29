import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 密度（Dense Layerの出力ニューロン数）
# 綾鷹AIを引き継いでとりあえず10にしておく（後から調整可）→　今回はRGBの3つの値を出力するので3
DENSE_SIZE = 3
# 画像サイズ
IMG_SIZE = 150
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE,3)
# 教師データ
X_TRAIN = []
Y_TRAIN = []
# テストデータ
X_TEST = []
Y_TEST = []

# パス定義
current_dir = os.getcwd()
# テストデータ保存先
TRAIN_TEST_DATA = os.path.join(current_dir, 'train_test_data/data.npy')
# data.npyが存在しなかったら作成
file_path_obj = Path(TRAIN_TEST_DATA)
if not file_path_obj.exists():
    file_path_obj.touch()
# モデル保存先
MODEL_ROOT_DIR = os.path.join(current_dir, 'models/')
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)


# ----- モデル構築 ----- #
model = keras.models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=INPUT_SHAPE))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation="relu"))
#活性化関数(activation)はsigmoidからreluに変更
model.add(layers.Dense(DENSE_SIZE,activation="relu"))

#モデル構成の確認
model.summary()
# ----- /モデル構築 ----- #


# ----- モデルコンパイル ----- #
# 損失関数(loss)はbinaryからmseに変更
model.compile(loss="mse",
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])
# ----- /モデル構築 ----- #


# ----- モデル学習 ----- #
# 教師データとテストデータを読み込む
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = np.load(TRAIN_TEST_DATA, allow_pickle=True)
model = model.fit(X_TRAIN,
                  Y_TRAIN,
                  epochs=100,
                  batch_size=32,
                  validation_data=(X_TEST, Y_TEST))
# ----- /モデル学習 ----- #


# ----- 学習結果プロット ----- #
# epochごとに学習結果がプロットされる
# loss: training loss 損失関数（今回はMSE）で計算した、機械学習モデルが予測した結果と実際の値のズレのこと
# acc: training accuracy
# val_loss: validation loss
# val_acc: validation accuracy
acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

# 学習結果をグラフにしたものをpngファイルにして保存
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(MODEL_ROOT_DIR, 'Training_and_validation_accuracy.png'))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(MODEL_ROOT_DIR, 'Training_and_validation_loss.png'))
# ----- /学習結果プロット ----- #


# ----- モデル保存 ----- #
# モデル保存
json_string = model.model.to_json()
open(os.path.join(MODEL_ROOT_DIR, 'model_predict.json'), 'w').write(json_string)

#重み保存
model.model.save_weights(os.path.join(MODEL_ROOT_DIR, 'model_predict.hdf5'))
# ----- /モデル保存 ----- #
