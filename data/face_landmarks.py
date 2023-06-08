import dlib
from imutils import face_utils
import cv2

# パスの定義
#カレントディレクトリの取得
current_dir = os.getcwd()
#画像のルートディレクトリ
img_root_dir = os.path.join(current_dir, 'images/')
#face_cutディレクトリへのパス
original_imgs_dir = os.path.join(img_root_dir, 'face_cut')
#ランドマーク検出した画像の保存先へのパス
result_imgs_dir = os.path.join(img_root_dir, 'face_landmarks')
os.makedirs(result_imgs__dir, exist_ok=True)

# 顔検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()

# 顔のランドマーク検出ツールの呼び出し
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)

# 検出対象の画像の呼び込み
img = cv2.imread('Girl.bmp')
# 処理高速化のためグレースケール化(任意)
img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------------------
# 2.顔のランドマーク検出
# --------------------------------
# 顔検出
# ※2番めの引数はupsampleの回数。基本的に1回で十分。
faces = face_detector(img_gry, 1)

# 検出した全顔に対して処理
for face in faces:
    # 顔のランドマーク検出
    landmark = face_predictor(img_gry, face)
    # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
    landmark = face_utils.shape_to_np(landmark)

    # ランドマーク描画
    for (i, (x, y)) in enumerate(landmark):
        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

# --------------------------------
# 3.結果表示
# --------------------------------
cv2.imshow('sample', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

