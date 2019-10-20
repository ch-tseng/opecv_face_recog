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
from libFaces import OBJTracking

th_score = 0.45
draw_face_box = True
video_file = "videos/news1.mp4"
cam_id = 0
webcam_size = (1024, 768)
cam_rotate = 0
flip_vertical = False
flip_horizontal = False
frame_display_size = (800, 600)
interval_frame_reRecognize = 6

write_output = True
output_video_path = "output_recog.avi"
video_size = (640, 480)  #x,y
video_rate = 22.0

source_photos_path = "employee_photos/"
add_employee_photos = "more_employee_photos"
dataset_path = "employee_embs/"
image_types = (".jpg", ".png", ".jpeg")
landmark_dat = "models/shape_predictor_5_face_landmarks.dat"

LM = facial(landmark_dat)
RG = faceRecognizer("models/openface.nn4.small2.v1.t7")

#如果尚沒有embs dataset，則從原始相片來建立
if not os.path.exists(dataset_path) and os.path.exists(source_photos_path):
    RG.make_embs(source_photos_path, dataset_path, landmark_dat, image_types)

#加入更多其它的相片
RG.make_embs(add_employee_photos, dataset_path, landmark_dat, image_types)
#載入embs dataset到memory
RG.load_embs_memory(dataset_path)

CAMERA = webCam(id=cam_id, videofile=video_file, size=webcam_size)
if(CAMERA.working() is False):
    print("webcam cannot work.")
    sys.exit()

CAMERA.set_record(outputfile="output_tsai.avi", video_rate=26)

def exit_app():
    print("End.....")
    CAMERA.stop_record()
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

target_in_frame = False
re_recognize = True
frameID = 0
last_Known_counts = 0
display_name = None
OB_TRACK = None

if __name__ == '__main__':
    hasFrame, frame_screen, frame_org = \
        CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=(frame_display_size[0], frame_display_size[1]))

    while hasFrame:
        bbox_success, bbox_boxes, bbox_names = [], [], []


        if(re_recognize is True):
            print("----------------> Re recofnize.")
            org_faces, aligned_faces, bbox_faces = LM.align_face(frame_org, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=1, maxOnly=False)
            for box in bbox_faces:
                bbox_boxes.append(box)
                bbox_success.append(True)

            Known_bbox = []
            Known_names = []

            for i, bbox in enumerate(bbox_faces):
                (name, score) = RG.verify_face(aligned_faces[i])
                print(name, score)

                if(score<th_score):
                    this_name = name
                    print_name = True
                    Known_bbox.append(bbox)
                    Known_names.append(name)

            OB_TRACK = OBJTracking()
            OB_TRACK.setROIs(frame_org, Known_bbox, "KCF")
            re_recognize = False

        if(OB_TRACK is not None):
            tracking_bbox, tracking_names = [], []
            print("Tracking......")
            print("    ", Known_names, Known_bbox)
            (success, roi_boxes) = OB_TRACK.trackROI(frame_org)
            print("    ", success, roi_boxes)
            for i, bbox in enumerate(roi_boxes):
                if(success is True):
                    tracking_bbox.append(roi_boxes)
                    tracking_names.append(Known_names[i])
                    color = OB_TRACK.roi_colors[i]
                    if(draw_face_box is True):
                        cv2.rectangle(frame_org, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (color[0],color[1],color[2]), 2)

                    this_name = Known_names[i]
                    display_name = this_name.split("_")
                    font_size = bbox[2]/64
                    frame_org = printText(display_name[1], frame_org, color=(color[0],color[1],color[2],0), size=font_size, pos=(bbox[0]+15,int(bbox[1]-(bbox[3]/2))), type="Chinese")


        if(len(tracking_bbox)==0):
            re_recognize = True

        cv2.imshow("FRAME", frame_org)
        CAMERA.write_video(frame_org)
        key = cv2.waitKey(1)
        if(key==113):
            exit_app()

        hasFrame, frame_screen, frame_org = \
            CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=(frame_display_size[0], frame_display_size[1]))

        frameID += 1

