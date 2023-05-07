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
def crop_face(path, write_path):
    img = cv2.imread(path)
    faces = faceDetector(img, 0)
    if len(faces) > 0:
    for i in range(0, len(faces)):      
        face_img = img[int(faces[i].top()):int(faces[i].bottom()),int(faces[i].left()):int(faces[i].right())]

files = glob.glob(os.path.join(original_imgs_dir, '*.jpeg'))
for i, file in enumerate(files):
    crop_face(original_imgs_dir, face_cut_dir)
    cv2.imwrite(write_path, face_img)