import os
import datetime
import pickle
import cv2
import numpy as np
import face_recognition
from PIL import Image
import streamlit as st
from deepface import DeepFace
import util
import sys

sys.path.append("Silent-Face-Anti-Spoofing")
from test1 import test

# Initialize Streamlit app
st.title("Face Recognition App")
db_dir = './db'
log_path = './log.txt'


@st.cache_data
def load_db():
    db = {}
    filenames = os.listdir(db_dir)
    for filename in filenames:
        path_ = os.path.join(db_dir, filename)
        with open(path_, 'rb') as file:
            embeddings = pickle.load(file)
            db[filename[:-7]] = embeddings
    return db


db = load_db()

# Session state to manage UI state
if 'page' not in st.session_state:
    st.session_state.page = 'main'

def show_main_page():
    # Webcam capture
    image = st.camera_input("Take a picture")

    if image:
        img = np.array(Image.open(image))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Anti-Spoofing
        label = test(
            image=img_rgb,
            model_dir='./Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
            device_id=0
        )

        if label == 1:
            if len(face_recognition.face_encodings(img_rgb)) == 0:
                st.write("No person detected")
            else:
                name = util.recognize(img_rgb, db)
                if name in ['unknown_person', 'no_persons_found']:
                    st.write("Unknown user. Please register if you have not or try again.")
                else:
                    result = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)
                    mood = result[0]['dominant_emotion']
                    st.write(f"Access granted. Welcome, {name}, you look {mood} today!")
                    with open(log_path, 'a') as f:
                        f.write(f'{name},{datetime.datetime.now()},in\n')
        else:
            name = util.recognize(img_rgb, db)
            if name in ['unknown_person', 'no_persons_found']:
                st.write("Hello unknown fake NPC. You shall not pass!")
            else:
                st.write(f"You think you are {name}? You shall not pass!")

            with open(log_path, 'a') as f:
                f.write(f'{name},{datetime.datetime.now()},in\n')

    if st.button("Register New User"):
        st.session_state.page = 'register'

def show_register_page():
    st.write("Register New User")
    new_user_name = st.text_input("Enter new user name")
    new_user_image = st.camera_input("Take a picture for registration")

    if st.button("Register New User"):
        if new_user_image and new_user_name:
            img = np.array(Image.open(new_user_image))
            embeddings = face_recognition.face_encodings(img)[0]
            with open(os.path.join(db_dir, f'{new_user_name}.pickle'), 'wb') as file:
                pickle.dump(embeddings, file)
            db[new_user_name] = embeddings
            st.write('User registered successfully!')
        else:
            st.write('Please provide a name and an image.')

    if st.button("Back"):
        st.session_state.page = 'main'

if st.session_state.page == 'main':
    show_main_page()
elif st.session_state.page == 'register':
    show_register_page()
