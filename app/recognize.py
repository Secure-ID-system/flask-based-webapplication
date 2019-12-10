import cv2
import numpy as np
import imutils
import pickle
import os
from pathlib import Path

def recognize(target, timestamp):
#    target = "/Users/syoon/Desktop/face_id/secure_face_id_web/app/static/img/IMG_20191204_015220.JPG"
    localized_addr = '/Users/syoon/Desktop/face_id/'
    dsPath = localized_addr + 'secure_face_id_web/app/static/dataset/'
    outPath = localized_addr + 'secure_face_id_web/app/output/'
    protoPath = localized_addr + 'secure_face_id_web/app/detection_model/deploy.prototxt'#os.path.sep.join(['detection_model','deploy.prototxt'])
    modelPath = localized_addr + 'secure_face_id_web/app/detection_model/res10_300x300_ssd_iter_140000.caffemodel'#os.path.sep.join(['detection_model','res10_300x300_ssd_iter_140000.caffemodel'])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embddPath = localized_addr + 'secure_face_id_web/app/openface_nn4.small2.v1.t7'#os.path.sep.join(['detection_model','deploy.prototxt'])
    embedder = cv2.dnn.readNetFromTorch(embddPath)
    recognizer = pickle.loads(open(outPath+'recognizer.pickle', "rb").read())
    le = pickle.loads(open(outPath+'le.pickle', "rb").read())
    
    image = cv2.imread(target.name)
    # print(image)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()
    attendants = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob through embedding model
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            crop = face.copy()
            if name is "unknown":
                dst = 'unknown/'+ timestamp + '.png'
            else:
                dst = name + '/' + timestamp + '.png'
            attendants.append('dataset/' + dst)
            dst = dsPath + dst
            cv2.imwrite(dst, crop)

            # draw the bounding box of the face along with the associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output image
    resPath = localized_addr + 'secure_face_id_web/app/static/result/'
    path = resPath + timestamp + '.png'
    cv2.imwrite(path, image.copy())
    # print(attendants)
    
    return path, attendants
    # dst = target.split("./images/")[1]
    # path = './result'
    # cv2.imwrite(os.path.join(path, dst), image.copy())
