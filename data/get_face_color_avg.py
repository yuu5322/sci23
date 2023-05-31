# python get_avg_face_color.py -h "ファイルのパス"

import numpy as np
import cv2
import os
import glob

# パスの定義
# 画像のルートディレクトリ
current_dir = os.getcwd()
img_root_dir = os.path.join(current_dir, 'images/')
# 顔だけを切り抜いた画像のパス
face_cut_dir = os.path.join(img_root_dir, 'face_cut')
# マスク後の画像の保存場所
masked_imgs_dir = os.path.join(img_root_dir, 'face_masked')
os.makedirs(masked_imgs_dir, exist_ok=True)

# 顔画像（カラー）の読み込み
color_files = glob.glob(os.path.join(face_cut_dir, '*.jpeg'))

for f in color_files:

    color_img = cv2.imread(f)

    try:
        # 処理している画像と同じ名前のマスク画像を読み込む
        mask_img_path = f.replace('/images/face_cut/', '/images/face_color/')
        mask_img = cv2.imread(mask_img_path)
        # そのままだと画像サイズが違ってエラーが出るので、グレースケールに変換
        #mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        # 画像をマスクして保存
        img_AND = cv2.bitwise_and(color_img, mask_img)
        file_name = f.replace('/images/face_cut/', '/images/face_masked/')
        cv2.imwrite(file_name, img_AND)

    except Exception as e:
        print(f)
        print(e)










# def parse_args():
#     parser = argparse.ArgumentParser(description="get averaged HSV color for image file(s)")
#     parser.add_argument("FILES", type=str, nargs="+", help="image file(s)")
#     return parser.parse_args()

# def proc_file(file):
#     img = cv2.imread(file)
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask_bin = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX) / 255
#     mask_bin = mask_bin.ravel().reshape(-1, 1)
#     mask_bin = np.repeat(mask_bin, img_hsv.shape[2], axis=1)
#     img_hsv = img_hsv.reshape(-1, 3)
#     sum_hsv = np.sum(img_hsv * mask_bin, axis=0)
#     sum_bin = np.sum(mask_bin, axis=0)
#     return sum_hsv / sum_bin

# def main(args):
#     avg = {}
#     for f in args.FILES:
#         avg[f] = proc_file(f)
#     for f in avg.keys():
#         print(f)
#         # print(avg[f])
#         val = avg[f]
#         for v in val:
#             print(v)

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)