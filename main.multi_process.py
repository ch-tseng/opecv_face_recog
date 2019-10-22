# -*- coding: UTF-8 -*-

import cv2
import imutils
import os, sys, time
import numpy as np
from sklearn.externals import joblib
from PIL import ImageFont, ImageDraw, Image
from libFaces import facial
from libFaces import webCam
from libFaces import faceRecognizer
from libFaces import OBJTracking
from random import randint
import multiprocessing as mp

th_score = 0.55
draw_face_box = True
video_file = "videos/C0007.MP4"
cam_id = 0
webcam_size = (1024, 768)
cam_rotate = 0
flip_vertical = False
flip_horizontal = False
frame_display_size = (800, 600)
interval_frame_reRecognize = 30

write_output = True
output_video_path = "output_ball1.avi"
video_size = (640, 480)  #x,y
video_rate = 22.0

source_photos_path = "employee_photos/"
add_employee_photos = "more_employee_photos"
dataset_path = "employee_embs/"
image_types = (".jpg", ".png", ".jpeg")
landmark_dat = "models/shape_predictor_5_face_landmarks.dat"

LM = facial(landmark_dat)
RG = faceRecognizer("models/openface.nn4.small2.v1.t7")
cpus = mp.cpu_count()

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
    pool_newface.close() # 關閉進程池，不再接受請求
    pool_newface.join() # 等待所有的子進程結束

    CAMERA.stop_record()
    pool_newface.close() # 關閉進程池，不再接受請求
    pool_newface.join() # 等待所有的子進程結束


    #print("Detect avg. time: {}/{}={}".format(LM.detect_time, LM.detect_count, round(LM.detect_time/LM.detect_count,5)) )
    #print("Landmark avg. time: {}/{}={}".format( LM.landmark_time, LM.landmark_count, round(LM.landmark_time/LM.landmark_count, 5) ))
    #print("Face alignment avg. time: {}/{}={}".format( LM.align_time,LM.align_count,round(LM.align_time/LM.align_count,5)  ))
    #print("EMBS time: {}/{}={}".format(embs_time, embs_count, round(embs_time/embs_count,5)))
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

def iou_bbox(boxA, boxB):

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

target_in_frame = False
re_recognize = True
frameID = 0
last_Known_counts = 0
display_name = None
OB_TRACK = None
last_Known_bbox = []

color = []
embs_time = 0
embs_count = 0
for i in range(30):
    color.append((randint(120, 255), randint(120, 255), randint(120, 255)))

#multi-process
#pool_list = []
#pool_newface = mp.Pool(processes = 1)

#def recog_face(frame, last_face_bboxes):
def recog_face(frame, last_face_bboxes):

    Known_bbox = []
    Known_names = []

    org_faces, aligned_faces, bbox_faces = LM.align_face(frame, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=1, maxOnly=False)
    for box in bbox_faces:
        #bbox_boxes.append(box)
        #bbox_success.append(True)

        #IOU
        if(len(last_face_bboxes)>0):
            for i, last_box in enumerate(last_face_bboxes):
                for ii, this_box in enumerate(bbox_faces):
                    iou_num = iou_bbox((last_box[0], last_box[1], last_box[0]+last_box[2], last_box[1]+last_box[3]),\
                        (this_box[0],this_box[1],this_box[0]+this_box[2],this_box[1]+this_box[3]))

                    #print("IOU: ", iou_num, ":", last_box, this_box)
                    if(iou_num>0.1):
                        Known_bbox.append(this_box)
                        Known_names.append(last_Known_names[i])
                        bbox_faces.pop(ii)
                        aligned_faces.pop(ii)
                        org_faces.pop(ii)

        for i, bbox in enumerate(bbox_faces):
            (name, score) = RG.verify_face(aligned_faces[i])
            #print(name, score)
            data_name = name.split('_')

            if(score<float(data_name[2])):
                this_name = name
                print_name = True
                Known_bbox.append(bbox)
                Known_names.append(name)

    return [Known_bbox, Known_names]

tracking_bbox = []
result_list = []
pool_list = []
Known_bbox, Known_names = [], []
if __name__ == '__main__':
    hasFrame, frame_screen, frame_org = \
        CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=(frame_display_size[0], frame_display_size[1]))

    while hasFrame:

        results = [xx.get() for xx in pool_list]
        if(len(results)>0):
            for result_detect in results:
                [Known_bbox, Known_names] = result_detect
                print("Pool reply:", Known_bbox, Known_names)

        if(re_recognize is True):

            #multi-process
            pool_list = []
            pool_newface = mp.Pool(processes = 1)

            #result_recogFace = pool_newface.apply_async(recog_face, (frame_org, last_Known_bbox,)) 
            #pool_result.append(result_recogFace)
            pool_list.append(pool_newface.apply_async(recog_face, args=(frame_org, last_Known_bbox,)))
            #results = [p.get() for p in result_recogFace]
            #print(results)
            #pool_newface.close() # 關閉進程池，不再接受請求
            #pool_newface.join() # 等待所有的子進程結束



            #print("----------------> Re recofnize.")
            #[Known_bbox, Known_names] = recog_face(frame_org, last_Known_bbox)

            results = [xx.get() for xx in pool_list]
            if(len(results)>0):
                for result_detect in results:
                    [Known_bbox, Known_names] = result_detect
                    print("Pool reply:", Known_bbox, Known_names)


            if(len(Known_bbox)>0):
                OB_TRACK = OBJTracking()
                OB_TRACK.setROIs(frame_org, Known_bbox, "KCF")
                re_recognize = False


        if(OB_TRACK is not None):
            tracking_bbox, tracking_names = [], []
            print("     Tracking......")
            #print("    ", Known_names, Known_bbox)
            (success, roi_boxes) = OB_TRACK.trackROI(frame_org)
            #print("    ", success, roi_boxes)


            if(len(roi_boxes)==len(Known_names)):
                for i, bbox in enumerate(roi_boxes):
                    if(success is True):
                        tracking_bbox.append(roi_boxes)
                        tracking_names.append(Known_names[i])
                        #color = OB_TRACK.roi_colors[i]
                        if(draw_face_box is True):
                            cv2.rectangle(frame_org, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),\
                                (color[i][0],color[i][1],color[i][2]), 2)

                        this_name = Known_names[i]
                        display_name = this_name.split("_")
                        font_size = bbox[2]/64
                        frame_org = printText(display_name[1], frame_org, color=(color[i][0],color[i][1],color[i][2],0), size=font_size, pos=(bbox[0]+15,int(bbox[1]-(bbox[3]/2))), type="Chinese")

                        last_Known_bbox = Known_bbox
                        last_Known_names = Known_names


        if(len(tracking_bbox)<2 and (frameID % interval_frame_reRecognize==0)):
            re_recognize = True
        else:
            re_recognize = False

        cv2.imshow("FRAME", imutils.resize(frame_org, width=frame_display_size[0]))
        CAMERA.write_video(frame_org)
        key = cv2.waitKey(1)
        if(key==113):
            exit_app()

        hasFrame, frame_screen, frame_org = \
            CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=(frame_display_size[0], frame_display_size[1]))

        '''
        if(frameID % interval_frame_reRecognize==0):
            #multi-process
            pool_list = []
            pool_newface = mp.Pool(processes = 1)

            #result_recogFace = pool_newface.apply_async(recog_face, (frame_org, last_Known_bbox,)) 
            #pool_result.append(result_recogFace)
            pool_list.append(pool_newface.apply_async(recog_face, args=(frame_org, last_Known_bbox,)))
            #results = [p.get() for p in result_recogFace]
            #print(results)
            pool_newface.close() # 關閉進程池，不再接受請求
            pool_newface.join() # 等待所有的子進程結束
        '''

        frameID += 1

    exit_app()