import cv2
import streamlit as st
import mediapipe as mp
from functions import *
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
model = load_model('model.h5')
actions = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'peace'])


def realTimeFeed():
    sequence = []
    threshold = 0.8
    sequence_length = 30

    stframe = st.empty()
    placeholder = st.empty()
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            _, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:sequence_length]
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(res)

                if np.max(res) > threshold:
                    label = actions[np.argmax(res)]
                    print(label)
                    with placeholder.container():
                        placeholder.text(f'Predicted Action: {label}')
                        sequence = []


            stframe.image(image, channels="BGR")


st.title("Real Time Feed")
realTimeFeed()
