from scipy.spatial.distance import cosine
import mtcnn
from keras.models import load_model
import pickle
import numpy as np
import cv2
from sklearn.preprocessing import Normalizer

def get_encode(face_encoder, face, size=(100,100), ):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode
def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)
l2_normalizer = Normalizer('l2')
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std
def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

encoder_model = 'data/model/facenet_keras.h5'
encodings_path = 'data/encodings/encodings.pkl'
test_img_path = 'data/test/test.jpeg'

recognition_t = 0.4
required_size = (160, 160)
encodings_dict = load_pickle(encodings_path)
face_detect = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)
img = cv2.imread(test_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detect.detect_faces(img_rgb)
for res in results:
    face, pt1, pt2 = get_face(img_rgb, res['box'])
    encode = get_encode(face_encoder, face, required_size)
    encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
    name = 'not recognize'
    distance = float("inf")

    for db_name, db_encode in encodings_dict.items():
        dist = cosine(db_encode, encode)
        if dist < recognition_t and dist < distance:
            name = db_name
            distance = dist
            print(name)
    if name == 'not recognize':
        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
        cv2.putText(img, name, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    else:
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img, name, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 200, 200), 2)
show(img)

