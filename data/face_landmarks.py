import dlib
from imutils import face_utils
import cv2
import os
import glob
import pandas as pd
import json

# パスの定義
#カレントディレクトリの取得
current_dir = os.getcwd()
# 顔検出器の学習済みモデルへのパス
predictor_path = os.path.join(current_dir, 'dlib/shape_predictor_68_face_landmarks.dat')
#画像のルートディレクトリ
img_root_dir = os.path.join(current_dir, 'images/')
#face_cutディレクトリへのパス
original_imgs_dir = os.path.join(img_root_dir, 'face_cut')
#ランドマーク検出した画像の保存先へのパス
result_imgs_dir = os.path.join(img_root_dir, 'face_landmarks')
os.makedirs(result_imgs_dir, exist_ok=True)

# 顔検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()

# 顔のランドマーク検出ツールの呼び出し
face_predictor = dlib.shape_predictor(predictor_path)

# 検出対象の画像の呼び込み
files = glob.glob(os.path.join(original_imgs_dir, '*.jpeg'))

#変数定義
output_dir = os.path.join(current_dir, 'csv/')
landmarks = []
#csvを出力する時の通し番号
file_number = 0

for f in files:
    img = cv2.imread(f)

    # 処理高速化のためグレースケール化(任意)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔のランドマーク検出

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
        #下唇の下側の座標（no.56~59）を逆にする（そうしないと変な形に切り抜かれるので）
        under_landmark = landmark[56:59]
        under_landmark = under_landmark[::-1]
        #リストにして詰める
        lip_landmarks.extend(under_landmark.tolist())
        lip_landmarks.extend(landmark[65:69].tolist())

        #ランドマークを配列に詰める（csv出力用）
        landmarks.append([file_number, f, json.dumps(lip_landmarks)])
        file_number = file_number + 1

        # ランドマーク描画
        for (i, (x, y)) in enumerate(lip_landmarks):
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

    # 生成した画像を保存
    # 元画像と同じ名前を付けて保存
    file_name = f.replace('/images/face_cut/', '/images/face_landmarks/')
    cv2.imwrite(file_name, img)

df = pd.DataFrame(landmarks, columns=['file_number','file_name', 'lip_landmarks'])
df.to_csv(output_dir + '/lip_landmarks.csv')