import dlib
import cv2

def crop_face(path, write_path, padding):

   faceDetector = dlib.get_frontal_face_detector()
   img = cv2.imread(path)
   faces = faceDetector(img, 0)

   if len(faces) > 0:
      for i in range(0, len(faces)):
         img_h, img_w, c = img.shape
         face_h = int(faces[i].bottom() - faces[i].top())
         face_w = int(faces[i].right() - faces[i].left())

         rect_top = int(faces[i].top()) - (face_h * padding)
         if rect_top < 0:
            rect_top = 0
         rect_bottom = int(faces[i].bottom()) + (face_h * padding)
         if rect_bottom > img_h:
            rect_bottom = img_h
         rect_left = int(faces[i].left()) - (face_w * padding)
         if rect_left < 0:
            rect_left = 0
         rect_right = int(faces[i].right()) + (face_w * padding)
         if rect_right > img_w:
            rect_right = img_w

         face_img = img[int(rect_top):int(rect_bottom),int(rect_left):int(rect_right)]
         cv2.imwrite(write_path, face_img)