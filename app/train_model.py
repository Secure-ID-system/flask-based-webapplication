from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

def train_model():
    outPath = '/Users/syoon/Desktop/face_id/secure_face_id_web/app/output/'
    data = pickle.loads(open(outPath+'embeddings.pickle', "rb").read())
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open(outPath+'recognizer.pickle', "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    
    f = open(outPath+'le.pickle', "wb")
    f.write(pickle.dumps(le))
    f.close()
    
    print("Train Done")
