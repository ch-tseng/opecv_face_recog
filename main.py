import cv2
import imutils
import os, time
from sklearn.externals import joblib
from libFaces import facial
from libFaces import faceRecognizer

source_photos_path = "employee_photos/"
dataset_path = "employee_embs/"
image_types = (".jpg", ".png", ".jpeg")

def make_dataset(photos_path, embsface_path):
    if not os.path.exists(dataset_path):
        os.makedirs(embsface_path)

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
                                embs = RG.get_embs(img)
                                print("    write embs to", output_embs_file)
                                joblib.dump(embs, output_embs_file)

                            print("    save finished.")
                            print("")


landmark_dat = "models/shape_predictor_5_face_landmarks.dat"

LM = facial(landmark_dat)
RG = faceRecognizer("models/openface.nn4.small2.v1.t7")

make_dataset(source_photos_path, dataset_path)

#org_faces, aligned_faces = LM.align_face(img_in, DOWNSAMPLE=1.0, DLIB_UPSAMPLE=1, maxOnly=True, drawRect=True)
#for i, img in enumerate(org_faces):
#    cv2.imshow("FACE_ORG", imutils.resize(img, height=120))
#    cv2.imshow("FACE_ALIGNED", imutils.resize(aligned_faces[i], height=120))
#    print(RG.get_embs(img))
#cv2.waitKey(0)

