import os
import cv2
import dlib
import numpy as np
# HSV range of lip color
HSV_MIN = np.array([0, 30, 70])
HSV_MAX = np.array([20, 150, 255])


def detect_faces(img):
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(img, 1)
    return faces

def proc_lip(img, debug=False):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    skin_mask = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX)
    if debug:
        print("DEBUG: output skin area image.")
        cv2.imwrite("skin_area.jpg", skin_mask)
    rev_mask_bin = 1 - skin_mask / 255
    rev_mask_bin = rev_mask_bin.ravel().reshape(-1, 1)
    rev_mask_bin = np.repeat(rev_mask_bin, img_hsv.shape[2], axis=1)
    img_hsv = img_hsv.reshape(-1, 3)
    sum_hsv = np.sum(img_hsv * rev_mask_bin, axis=0)
    sum_bin = np.sum(rev_mask_bin, axis=0)
    return sum_hsv / sum_bin

def proc_face(rgb_img, face_rect, shape_predictor, lip_points, debug=False):
    shapes = shape_predictor(rgb_img, face_rect)
    parts_pos = [[p.x, p.y] for p in shapes.parts()]
    lip_pos = np.array(parts_pos[lip_points[0] : (lip_points[1] + 1)]).T
    lip_img = rgb_img[
        lip_pos[1].min() : lip_pos[1].max(), lip_pos[0].min() : lip_pos[0].max()
    ]  # [y_min:y_max, x_min:x_max]
    if debug:
        print("DEBUG: output lip area image.")
        cv2.imwrite("lip_area.jpg", cv2.cvtColor(lip_img, cv2.COLOR_RGB2BGR))
    return proc_lip(lip_img, debug)

def proc_file(file, shape_predictor, lip_points):
    img = cv2.imread(file)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detect_faces(rgb_img)
    avg = []
    for f in faces:
        avg.append(proc_face(rgb_img, f, shape_predictor, lip_points, debug=args.debug))
    return avg

def main(args):
    avg = {}
    shape_predictor = dlib.shape_predictor(args.model[0])
    lip_points_str = args.lip_points[0].split(":")
    lip_points = (int(lip_points_str[0]), int(lip_points_str[1]))
    for file in args.FILES:
        avg[file] = proc_file(file, shape_predictor, lip_points)
    for file in avg.keys():
        print(file)
        for idx, val in enumerate(avg[file]):
            # print(f"Face ID:{idx} Averaged Lip Color in HSV:{val}")
            # print(val)
            for v in val:
                print(v)
