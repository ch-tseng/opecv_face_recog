# -*- coding: UTF-8 -*-

import cv2
import imutils
import os, time
import numpy as np
from sklearn.externals import joblib
from PIL import ImageFont, ImageDraw, Image
from libFaces import facial
from libFaces import webCam
from libFaces import faceRecognizer

th_score = 0.45
video_file = "videos/news1.mp4"
cam_id = 0
webcam_size = (1024, 768)
cam_rotate = 0
flip_vertical = False
flip_horizontal = False
frame_display_size = (1024, 768)

source_photos_path = "employee_photos/"
dataset_path = "employee_embs/"
image_types = (".jpg", ".png", ".jpeg")
landmark_dat = "models/shape_predictor_5_face_landmarks.dat"

LM = facial(landmark_dat)
RG = faceRecognizer("models/openface.nn4.small2.v1.t7")

#如果尚沒有embs dataset，則從原始相片來建立
if not os.path.exists(dataset_path) and os.path.exists(source_photos_path):
    RG.make_embs(source_photos_path, dataset_path, landmark_dat, image_types)

#載入embs dataset到memory
RG.load_embs_memory(dataset_path)

CAMERA = webCam(id=cam_id, videofile=video_file, size=webcam_size)
if(CAMERA.working() is False):
    print("webcam cannot work.")
    sys.exit()

def exit_app():
    print("End.....")
    sys.exit(0)

def printText(txt, bg, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
    (b,g,r,a) = color

    if(type=="English"):
        ## Use cv2.FONT_HERSHEY_XXX to write English.
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

    else:
        ## Use simsum.ttf to write Chinese.
        fontpath = "wt009.ttf"
        #print("TEST", txt)
        font = ImageFont.truetype(fontpath, int(size*20))
        img_pil = Image.fromarray(bg)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos,  txt, font = font, fill = (b, g, r, a))
        bg = np.array(img_pil)

    return bg

if __name__ == '__main__':
    hasFrame, frame_screen, frame_org = \
        CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=(frame_display_size[0], frame_display_size[1]))

    while hasFrame:
        org_faces, aligned_faces, bbox_faces = LM.align_face(frame_org, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=1, maxOnly=False)
        print(len(org_faces), len(aligned_faces), len(bbox_faces))

        for i, bbox in enumerate(bbox_faces):
            (name, score) = RG.verify_face(aligned_faces[i])
            if(score<th_score):
                print(name, score)
                cv2.rectangle(frame_org, bbox, (0,255,0), 2)
                frame_org = printText(name, frame_org, color=(0,255,255,0), size=1.8, pos=(bbox[0],bbox[1]-10), type="Chinese")

        cv2.imshow("FRAME", frame_org)
        key = cv2.waitKey(1)
        if(key==113):
            exit_app()

        hasFrame, frame_screen, frame_org = \
            CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=(frame_display_size[0], frame_display_size[1]))

