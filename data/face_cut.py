import cv2
import os
import glob

#顔検出器のインポート
import dlib
faceDetector = dlib.get_frontal_face_detector()

#画像のルートディレクトリ
img_root_dir = './images/'
#original画像のディレクトリへのパス
original_imgs_dir = os.path.join(img_root_dir, 'original')
#顔だけを切り抜いた画像の保存先へのパス
face_cut_dir = os.path.join(img_root_dir, 'face_cut')
os.makedirs(face_cut_dir, exist_ok=True)

#顔を切り抜く関数をdefで定義
def crop_face(path):
    img = cv2.imread(path)
    faces = faceDetector(img, 0)
    if len(faces) > 0:
        for i in range(0, len(faces)):# face_img には i 番目の顔のみが入る（他の顔の情報は無視される）
            face_img = img[int(faces[i].top()):int(faces[i].bottom()),int(faces[i].left()):int(faces[i].right())]
    return face_img

files = glob.glob(os.path.join(original_imgs_dir, '*.jpeg'))
for f in files:
    try:
        face_img = crop_face(f)
        file_name = f.replace('./images/original/', './images/face_cut/')
        cv2.imwrite(file_name, face_img)
    except Exception as e:
        print(f)
        print(e)
