# python get_avg_face_color.py -h "ファイルのパス"

import argparse
import numpy as np
import cv2
HSV_MIN = np.array([0, 30, 70])
HSV_MAX = np.array([20, 150, 255])

def parse_args():
    parser = argparse.ArgumentParser(description="get averaged HSV color for image file(s)")
    parser.add_argument("FILES", type=str, nargs="+", help="image file(s)")
    return parser.parse_args()

def proc_file(file):
    img = cv2.imread(file)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_bin = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX) / 255
    mask_bin = mask_bin.ravel().reshape(-1, 1)
    mask_bin = np.repeat(mask_bin, img_hsv.shape[2], axis=1)
    img_hsv = img_hsv.reshape(-1, 3)
    sum_hsv = np.sum(img_hsv * mask_bin, axis=0)
    sum_bin = np.sum(mask_bin, axis=0)
    return sum_hsv / sum_bin

def main(args):
    avg = {}
    for f in args.FILES:
        avg[f] = proc_file(f)
    for f in avg.keys():
        print(f)
        # print(avg[f])
        val = avg[f]
        for v in val:
            print(v)

if __name__ == "__main__":
    args = parse_args()
    main(args)