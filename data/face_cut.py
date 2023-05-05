import cv2
import os
import uuid
import glob

#カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
HAAR_FILE = "./OpenCV/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)

#face_cut_imgsディレクトリまでのパス
# imgs_dir_path = os.path.join(os.getcwd(),"data/face_cut_imgs/")
#サムネイル画像が入っているディレクトリまでのパス
# thumbnails_dir_path = os.path.join(os.getcwd(),"data/thumbnail_imgs/")

# 卒論の図で使う画像までのパス
example_img_path = /Users/yukahirose/Documents/ゼミ/卒論/graduation-thesis/data/thumbnail_imgs/2d83a6bc-9e60-4087-9d4a-10a7c0ba2f2b.jpeg

#data/thumbnail_imgs内の画像を全て読み込む
# files = glob.glob(str(thumbnails_dir_path) + str("*"))

# for f in files:

    #画像ファイルの読み込み
    # image_picture = f
img = cv2.imread(example_img_path)

    #グレースケールに変換する
img_g = cv2.imread(example_img_path,0)

    #カスケード型分類器を使用して画像ファイルから顔部分を検出する
face = cascade.detectMultiScale(img_g)

    #顔の座標を表示する
    # print(face)

    #顔部分を切り取る
for x,y,w,h in face:
    face_cut = img[y:y+h, x:x+w]

    # 白枠で顔を囲む
for x,y,w,h in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

    # face_cut_imgsフォルダが存在しなかった場合はフォルダを作る
    # if not os.path.exists(./):
    #     os.mkdir(dirname)

    #画像の出力
    # cv2.imwrite(imgs_dir_path + str(uuid.uuid4()) + str('.jpeg'), face_cut)
    # cv2.imwrite('face_rectangle.jpg', img)
cv2.imwrite(str('/Users/yukahirose/Documents/ゼミ/卒論/graduation-thesis/') + str(uuid.uuid4()) + str('.jpeg'), face_cut)
cv2.imwrite(str('/Users/yukahirose/Documents/ゼミ/卒論/graduation-thesis/') + str(uuid.uuid4()) + str('.jpeg'), img)