import cv2
import imutils
import os, time
from sklearn.externals import joblib
from libFaces import facial
from libFaces import faceRecognizer

source_photos_path = "employee_photos/"
dataset_path = "employee_embs/"
image_types = (".jpg", ".png", ".jpeg")
landmark_dat = "models/shape_predictor_5_face_landmarks.dat"

#LM = facial(landmark_dat)
RG = faceRecognizer("models/openface.nn4.small2.v1.t7")

#如果尚沒有embs dataset，則從原始相片來建立
if not os.path.exists(dataset_path) and os.path.exists(source_photos_path):
    RG.make_embs(source_photos_path, dataset_path, landmark_dat, image_types)

#載入embs dataset到memory
RG.load_embs_memory(dataset_path)

#org_faces, aligned_faces = LM.align_face(img_in, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=1, maxOnly=True, drawRect=True)
#for i, img in enumerate(org_faces):
#    cv2.imshow("FACE_ORG", imutils.resize(img, height=120))
#    cv2.imshow("FACE_ALIGNED", imutils.resize(aligned_faces[i], height=120))
#    print(RG.get_embs(img))
#cv2.waitKey(0)

