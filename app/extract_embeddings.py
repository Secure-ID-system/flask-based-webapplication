from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

def extract_embeddings():
    localized_addr = '/Users/syoon/Desktop/face_id/'
    protoPath = localized_addr + 'secure_face_id_web/app/detection_model/deploy.prototxt'#os.path.sep.join(['detection_model','deploy.prototxt'])
    modelPath = localized_addr + 'secure_face_id_web/app/detection_model/res10_300x300_ssd_iter_140000.caffemodel'#os.path.sep.join(['detection_model','res10_300x300_ssd_iter_140000.caffemodel'])
    embddPath = localized_addr + 'secure_face_id_web/app/openface_nn4.small2.v1.t7'#os.path.sep.join(['detection_model','deploy.prototxt'])
    dsPath = localized_addr + 'secure_face_id_web/app/static/dataset'
    outPath = localized_addr + 'secure_face_id_web/app/output/'
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embedder = cv2.dnn.readNetFromTorch(embddPath)
    given = list(paths.list_images(dsPath))

    knownEmbeddings = []
    knownNames = []
    total = 0

    for (i, imagePath) in enumerate(given):
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    print("Extracted embeddings of {}".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(outPath + 'embeddings.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()
