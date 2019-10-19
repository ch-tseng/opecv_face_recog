import sys
import os, time
import glob
import cv2
import imutils
import numpy as np
import math
import dlib
from sklearn.externals import joblib
import _pickle as cPickle

class faceRecognizer:
    def __init__(self, modelPath):
        self.model = cv2.dnn.readNetFromTorch(modelPath)
        self.recMean = [0,0,0]
        self.recSize = (96, 96)
        self.recScale = 1/255.0

    def setDataset(self, folderPath):
        subfolders = []
        for x in os.listdir(faceDatasetFolder):
            xpath = os.path.join(faceDatasetFolder, x)
            if os.path.isdir(xpath):
                subfolders.append(xpath)

        nameLabelMap = {}
        labels = []
        imagePaths = []
        for i, subfolder in enumerate(subfolders):
            for x in os.listdir(subfolder):
                xpath = os.path.join(subfolder, x)
                if x.endswith('jpg'):
                    imagePaths.append(xpath)
                    labels.append(i)
                    nameLabelMap[xpath] = subfolder.split('/')[-1]

        return nameLabelMap

    def get_embs(self, alignedFace):
        recModel = self.model
        blob = cv2.dnn.blobFromImage(alignedFace, self.recScale, self.recSize, self.recMean, False, False)
        recModel.setInput(blob)
        faceDescriptor = recModel.forward()

        return faceDescriptor

    def make_embs(self, photos_path, embsface_path, landmark_5_dat_path, image_types=(".jpg", ".jpeg", ".png")):
        if not os.path.exists(embsface_path):
            os.makedirs(embsface_path)

        LM = facial(landmark_5_dat_path)

        for id, idName in enumerate(os.listdir(photos_path)):
            idName_path = os.path.join(photos_path, idName)
            if(os.path.isdir(idName_path)):
                try:
                    uid, uname = idName.split('_')

                except:
                    print(idName, "folder format error!")
                    continue

                for idCount, imgfile in enumerate(os.listdir(idName_path)):
                    img_path = os.path.join(idName_path, imgfile)

                    if(os.path.isfile(img_path)):
                        filename, file_extension = os.path.splitext(imgfile)

                        if(file_extension.lower() in image_types):
                            print("{} processing {}".format(idCount, img_path))
                            img_in = cv2.imread(img_path)
                            start_time = time.time()
                            org_faces, aligned_faces = LM.align_face(img_in, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=2, maxOnly=False, drawRect=False)
                            #cv2.imshow("aligned_faces", aligned_faces)
                            time_used = round(time.time() - start_time, 3)
                            print("    time used:", time_used)

                            if(len(aligned_faces)>0):
                                embs_idname_path = os.path.join(embsface_path, idName)
                                if not os.path.exists( embs_idname_path ):
                                    os.makedirs(embs_idname_path)

                                for i, img in enumerate(aligned_faces):
                                    output_embs_file = os.path.join(embs_idname_path, filename+"_embs_"+str(i)+".embs")
                                    cv2.imwrite(os.path.join(embs_idname_path, filename+"_face_"+str(i)+".jpg"), img)
                                    embs = self.get_embs(img)
                                    print("    write embs to", output_embs_file)
                                    joblib.dump(embs, output_embs_file)

                                print("    save finished.")
                                print("")

    def load_embs_memory(self, embs_path):
        all_embs = []

        for id, file in enumerate(os.listdir(embs_path)):
            file_path = os.path.join(embs_path, file)

            if(os.path.isfile(file_path)):
                filename, file_extension = os.path.splitext(file)
                if(file_extension.lower() == '.embs'):
                     try:
                          all_embs.append((filename, joblib.load(file_path)) )

                     except:
                          logging.info("[error except] load_embs_memory() --> Error on loading embs: {}".format(file_path))
                          pass

        self.embs_dataset = all_embs

    def compare_embs(self, sb_emb):
        min_score = 999
        data_people = "999999_unknow"
        for id, (nameData, emb) in enumerate(embsALL):
            cac_emb = calc_dist(emb, sb_emb)
            if(cac_emb is not None):
                #print("emb diff = ", cac_emb, nameData)
                if(cac_emb<min_score):
                    min_score = cac_emb
                    data_people = nameData

        #print(data_people, min_score)
        return (data_people, min_score)


class faceDetect_Cascade:
    def __init__(self, xmlPath):
        self.face_cascade = cv2.CascadeClassifier(xml)
        self.cascade_scale = 1.1
        self.cascade_neighbors = 6
        self.minFaceSize = (30,30)

    def getFaces(self, img, DOWNSAMPLE=1.0):
        if(DOWNSAMPLE_RATIO>1.0):
            imSmall = cv2.resize(img,None, \
                fx=1.0/DOWNSAMPLE, \
                fy=1.0/DOWNSAMPLE, \
                interpolation = cv2.INTER_LINEAR)
        else:
            imSmall = img.copy()

        CASCADE = self.face_cascade
        gray = cv2.cvtColor(imSmall, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray,
            scaleFactor = self.cascade_scale,
            minNeighbors = self.cascade_neighbors,
            minSize = self.minFaceSize,
            flags = self.cv2.CASCADE_SCALE_IMAGE
        )

        bboxes = []
        for (x,y,w,h) in faces:
            if(w>minFaceSize[0] and h>minFaceSize[1]):
                bboxes.append((x*DOWNSAMPLE, y*DOWNSAMPLE, w*DOWNSAMPLE, h*DOWNSAMPLE))

        return bboxes

class facial:
    def __init__(self, modelPath):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(modelPath)

    def detect_face(self, img, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=5, maxOnly=False):
        if(int(DOWNSAMPLE)!=1):
            imSmall = cv2.resize(img,None, \
                fx=1.0/DOWNSAMPLE, \
                fy=1.0/DOWNSAMPLE, \
                interpolation = cv2.INTER_LINEAR)
        else:
            imSmall = img.copy()

        detector = self.detector
        dets = detector(imSmall, DLIB_UPSAMPLE)
        #print("dets:", dets)
        bbox = []
        dects_resize = []
        maxArea = 0
        for k, d in enumerate(dets):
            dd = dlib.rectangle(left=int(d.left()*DOWNSAMPLE), right=int(d.right()*DOWNSAMPLE), \
                top=int(d.top()*DOWNSAMPLE), bottom=int(d.bottom()*DOWNSAMPLE))

            if(maxOnly is True):
                if((dd.right()-dd.left()) * (dd.bottom()-dd.top()) > maxArea):
                    maxArea = (dd.right()-dd.left()) * (dd.bottom()-dd.top())
                    #dd_max = dlib.rectangle(left=int(d.left()*DOWNSAMPLE), right=int(d.right()*DOWNSAMPLE), \
                    #    top=int(d.top()*DOWNSAMPLE), bottom=int(d.bottom()*DOWNSAMPLE))
                    dd_max = dd
            else:
                bbox.append((dd.left(), dd.top(), dd.right()-dd.left(), dd.bottom()-dd.top())) 
                #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #    k, dd.left(), dd.top(), dd.right(), dd.bottom()))
                dects_resize.append(dd)

        if(len(dets)>0 and maxOnly is True):
            bbox.append((dd_max.left(), dd_max.top(), dd_max.right()-dd_max.left(), dd_max.bottom()-dd_max.top()))
            dects_resize.append(dd_max)

        return dects_resize, bbox

    def renderFace(self, im, landmarks, color=(0, 255, 0), radius=6):
        for p in landmarks:
            cv2.circle(im, (p[0], p[1]), radius, color, -1)


    def landmark_face(self, img, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=1, maxOnly=False, drawRect=True):
        predictor = self.predictor
        '''
        if(int(DOWNSAMPLE)!=1):
            imSmall = cv2.resize(img,None, \
                fx=1.0/DOWNSAMPLE, \
                fy=1.0/DOWNSAMPLE, \
                interpolation = cv2.INTER_LINEAR)
        else:
            imSmall = img.copy()
        '''
        #gray = cv2.cvtColor(imSmall, cv2.COLOR_BGR2GRAY)
        faces_rects, bboxes = self.detect_face(img, DOWNSAMPLE=DOWNSAMPLE, DLIB_UPSAMPLE=DLIB_UPSAMPLE, maxOnly=maxOnly)
        #print("Faces:", len(faces_rects))
        faces_landmarks = []
        #bboxes = []
        for (k, d) in enumerate(faces_rects):
            landmarks = []
            #print(d)
            #bbox = (int(d.left() * DOWNSAMPLE), int(d.top() * DOWNSAMPLE), \
            #    int((d.right()-d.left()) * DOWNSAMPLE), int((d.bottom()-d.top()) * DOWNSAMPLE))
            #bboxes.append(bbox)
            shape = predictor(img, d)
            for p in shape.parts():
                #landmarks.append((int(p.x * DOWNSAMPLE), int(p.y * DOWNSAMPLE)))                
                landmarks.append((p.x, p.y))

            if(drawRect is True):
                cv2.rectangle(img, (bboxes[k][0], bboxes[k][1]), (bboxes[k][0]+bboxes[k][2], bboxes[k][1]+bboxes[k][3]), (0,255,0), 2)
                self.renderFace(img, landmarks )

            faces_landmarks.append( (bboxes[k], landmarks) )

        return img, faces_landmarks, bboxes


    def angle_2points(self, p1, p2):
        r_angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        rotate_angle = r_angle * 180 / math.pi

        return rotate_angle

    def area_expand(self, bbox, ratio):
        ew = int(bbox[3] * ratio)
        eh = int(bbox[2] * ratio)
        nx = int(bbox[0] - ((ew - bbox[2]) / 2))
        ny = int(bbox[1] - ((eh - bbox[3]) / 2))
        if(nx<0):
            nx = 0
        if(ny<0):
            ny = 0

        return (nx,ny,ew,eh)

    def align_face(self, img, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=1, maxOnly=False, drawRect=True): 
        faces_org, faces_aligned = [], []
        image, face_landmarks, faces = self.landmark_face(img.copy(), DOWNSAMPLE=DOWNSAMPLE, DLIB_UPSAMPLE=DLIB_UPSAMPLE, \
            maxOnly=maxOnly, drawRect=drawRect)

        if(len(face_landmarks)>0):
            for (bbox, landmarks) in face_landmarks:
                x_a1, x_a2 = landmarks[0][0], landmarks[1][0] 
                y_a1, y_a2 = landmarks[0][1], landmarks[1][1]
                (cx1, cy1) = ( int((x_a1+x_a2)/2), int((y_a1+y_a2)/2) )

                x_b1, x_b2 = landmarks[2][0], landmarks[3][0] 
                y_b1, y_b2 = landmarks[2][1], landmarks[3][1]
                (cx2, cy2) = ( int((x_b1+x_b2)/2), int((y_b1+y_b2)/2) )

                landmarks.append((cx1, cy1))
                landmarks.append((cx2, cy2))

                face_area_org = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                faces_org.append(face_area_org)

                rotate_angle = self.angle_2points((cx1,cy1), (cx2,cy2))
                (nx, ny, ew, eh) = self.area_expand(bbox, 1.5)
                face_area_expand_org = img[ny:(ny+eh), nx:(nx+ew)]

                face_area_align_org = imutils.rotate_bound(face_area_expand_org, -rotate_angle)
                _, face_aligned = self.detect_face(face_area_align_org, DOWNSAMPLE=DOWNSAMPLE, DLIB_UPSAMPLE=DLIB_UPSAMPLE, \
                    maxOnly=True)

                if(len(face_aligned)>0):
                    xx, yy, ww, hh = face_aligned[0][0], face_aligned[0][1], face_aligned[0][2], face_aligned[0][3]
                    face_area_align_crop = face_area_align_org[yy:yy+hh, xx:xx+ww]
                    faces_aligned.append(face_area_align_crop)

        return faces_org, faces_aligned

    def draw_point(self, img, p, color):
        cv2.circle(img, p, 2, color, 0)

    def rect_contains(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False

        return True

    def draw_delaunay(self, img, subdiv, delaunary_color, drawRect):
        d_points = []
        triangleList = subdiv.getTriangleList()
        triangles_list = np.array(triangleList, dtype=np.int32)
        size = img.shape
        r = (0, 0, size[1], size[0])
    
        for t in triangles_list:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
    
            if (drawRect is True) and (self.rect_contains(r, pt1) and self.rect_contains(r, pt2) and self.rect_contains(r, pt3)):
                cv2.line(img, pt1, pt2, delaunary_color, 1)
                cv2.line(img, pt2, pt3, delaunary_color, 1)
                cv2.line(img, pt3, pt1, delaunary_color, 1)

            d_points.append([pt1,pt2,pt3])

        return d_points

    def face_delaunay(self, img, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=1, maxOnly=False, drawRect=True):
        delaunay_color = (255, 255, 255)
        points_color = (0, 0, 255)

        _, landmarks, faces = self.landmark_face(img.copy(), DOWNSAMPLE=DOWNSAMPLE, DLIB_UPSAMPLE=DLIB_UPSAMPLE, \
            maxOnly=maxOnly, drawRect=False)

        delaunay_faces = []
        size = img.shape

        for i, flandmark in enumerate(landmarks):
            rect = (0, 0, size[1], size[0])
            subdiv = cv2.Subdiv2D(rect)
            points = []
            #print("TEST:", flandmark)

            face_landmarks = flandmark[1]

            for (x,y) in face_landmarks:
                points.append((x, y))

                for pid, p in enumerate(points):
                    subdiv.insert(p)

            points_delaunay = self.draw_delaunay( img, subdiv, (255, 255, 255), drawRect=drawRect )
            #f = open("from.points", "w")
            lines = []
            for points_3 in points_delaunay:
                d_line = "{},{},{},{},{},{}|{},{},{}".\
                    format(int(points_3[0][0]), int(points_3[0][1]), \
                    int(points_3[1][0]), int(points_3[1][1]), int(points_3[2][0]), int(points_3[2][1]), \
                    points.index((int(points_3[0][0]), int(points_3[0][1]))), \
                    points.index((int(points_3[1][0]), int(points_3[1][1]))), \
                    points.index((int(points_3[2][0]), int(points_3[2][1]))) )
                lines.append(d_line)

            delaunay_faces.append((faces[i], face_landmarks, lines))
            #f.close()
            if(drawRect is True):
                for p in points :
                    self.draw_point(img, p, (0,0,255))

        return img, delaunay_faces

    def tri_transform(self, img1, img2, tri1, tri2) :
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(tri1)
        r2 = cv2.boundingRect(tri2)
    
        # Crop input image
        img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        # Offset points by left top corner of the respective rectangles
        tri1Cropped = []
        tri2Cropped = []

        for i in range(0, 3):
            tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
            tri2Cropped.append(((tri2[0][i][0] - r2[0]),(tri2[0][i][1] - r2[1])))

        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
    
        # Apply the Affine Transform just found to the src image
        #img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, \
        #    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
        img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR )
        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);

        img2Cropped = img2Cropped * mask
    
        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

        return img2

    def face_morphing(self, img_target, img_from_many, trans_type=0, DOWNSAMPLE=1.0, \
            DLIB_UPSAMPLE=1, maxOnly=True, drawRect=False):

        img_render_target, facial_data_target = self.face_delaunay(img_target, DOWNSAMPLE=DOWNSAMPLE, DLIB_UPSAMPLE=DLIB_UPSAMPLE, \
            maxOnly=True, drawRect=drawRect)
        target_landmarks = facial_data_target[0][1]
        print("TARGET LANDMARKS:", target_landmarks)

        img_render_in, facial_data_in = self.face_delaunay(img_from_many, DOWNSAMPLE=DOWNSAMPLE, DLIB_UPSAMPLE=DLIB_UPSAMPLE, \
            maxOnly=maxOnly, drawRect=drawRect)
        print("IN LANDMARKS:", facial_data_in)

        output_imgs = []
        facial_data = []
        for i, (face_bbox, landmark, delaunay)  in enumerate(facial_data_in):
            from_tri_points = []
            to_tri_points = []

            for line in delaunay:
                line0, line1 = line.split("|")
                xy = line0.split(',')
                ixy = line1.split(',')
                from_tri_points.append([ [int(xy[0]),int(xy[1])], [int(xy[2]),int(xy[3])], [int(xy[4]), int(xy[5])]  ])
                to_tri_points.append([ [target_landmarks[int(ixy[0])][0], target_landmarks[int(ixy[0])][1]], \
                    [target_landmarks[int(ixy[1])][0], target_landmarks[int(ixy[1])][1]],\
                    [target_landmarks[int(ixy[2])][0], target_landmarks[int(ixy[2])][1]]  ])

                for ii, to_tri in enumerate(to_tri_points):
                    #print(to_tri)
                    #print(from_tri_points[i])
                    # Input triangle
                    triIn = np.float32([from_tri_points[ii]])
                    # Output triangle
                    triOut = np.float32([to_tri])
  
                    # Warp all pixels inside input triangle to output triangle
                    if(trans_type==0): # Many to 1: Many faces to Lin's face, and put on Lin's image
                        img_final = self.tri_transform(img_from_many, img_target, triIn, triOut)
                    else:
                        img_from_many = self.tri_transform(img_target, img_from_many, triOut, triIn)

                    # Draw triangle using this color
                    color = (255, 150, 0)

                    # Draw triangles in input and output images.
                    #cv2.polylines(img_from, triIn.astype(int), True, color, 0, cv2.LINE_AA)

            if(trans_type==0):
                output_imgs.append(img_final.copy())
            else:
                output_imgs.append(img_from_many.copy())
            #cv2.imshow("TEST", img_final)
            #cv2.waitKey(0)
            facial_data.append((face_bbox, landmark, delaunay))

        return output_imgs, facial_data
