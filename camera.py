from scipy.spatial.distance import cosine
import mtcnn
from keras.models import load_model
import pickle
import numpy as np
import cv2
from sklearn.preprocessing import Normalizer


def get_encode(face_encoder, face, size):
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
def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict
def recognize(img,detector,encoder,encoding_dict,recognition_t=0.6,confidence_t=0.99,required_size=(160, 160), ):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]

        name = 'not recognize'
        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'not recognize':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 255), 2)
            cv2.putText(img, name + f'{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img


if __name__ == '__main__':
    encoder_model = 'data/model/facenet_keras.h5'
    encodings_path = 'data/encodings/encodings.pkl'

    face_detector = mtcnn.MTCNN()
    face_encoder = load_model(encoder_model)
    encoding_dict = load_pickle(encodings_path)

    vc = cv2.VideoCapture(0)
    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            print("no frame:(")
            break
        frame = recognize(frame, face_detector, face_encoder, encoding_dict)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
