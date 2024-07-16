from pveagle import EagleProfile
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import cv2
import numpy as np
import face_recognition
from PIL import Image
import gradio as gr
from deepface import DeepFace
import pveagle
from pvrecorder import PvRecorder
import sys
import time

sys.path.append("Silent-Face-Anti-Spoofing")
from test1 import test  # from test1.py in the Silent-Face-Anti-Spoofing folder
# gr.themes.builder()

uri = "mongodb+srv://longmw2001:g2XoE2TTYOKjGVFg@facerecognition.hwcxpnq.mongodb.net/?retryWrites=true&w=majority&appName=FaceRecognition"
client = MongoClient(uri, server_api=ServerApi('1'))
collection = client['users']['faces']
collection.create_index("username", unique=True)

# Load all embeddings into memory
user_embeddings = {}

def load_embeddings():
    global user_embeddings
    user_embeddings = {}
    for user in collection.find():
        stats = {}
        username = user['username']
        embedding = np.array(user['embedding'])
        voice_profile = user.get('voice_profile', None)
        if voice_profile:
            voice_profile = EagleProfile.from_bytes(voice_profile)
        stats['face'] = embedding
        stats['speaker'] = voice_profile
        user_embeddings[username] = stats
        print(username)


load_embeddings()

access_key = "/cxYMcgSAw1EMF0XPQT2tXUKHoHqXSCvByLxTjtmMjHfwgkN/ypnag=="


state = 1
usersname=""

def recognize_user(image):
    global state, usersname
    if state == 1:
        img = np.array(image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(face_recognition.face_encodings(img_rgb)) == 0:
            return "No person detected"

        # Anti-Spoofing
        label = test(
            image=img_rgb,
            model_dir='./Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
            device_id=0
        )

        if label == 1:
            name = recognize(img_rgb)
            if name in ['unknown_person', 'no_persons_found']:
                return "Unknown user. Please register if you have not or try again."
            else:
                state = 2
                result = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)
                mood = result[0]['dominant_emotion']
                usersname = name
                if name.endswith('_0001') or name.endswith('_0002') or name.endswith('_0003'):
                    name = name[:-5]
                return f"Hello {name}, you look {mood} today! Verify your voice to enter."
        else:
            name = recognize(img_rgb)
            if name in ['unknown_person', 'no_persons_found']:
                return "Hello unknown fake NPC. You shall not pass!"
            else:
                if name.endswith('_0001') or name.endswith('_0002') or name.endswith('_0003'):
                    name = name[:-5]
                return f"You think you are {name}? You shall not pass!"
    else:
        state = 1
        speaker_profile = user_embeddings[usersname]['speaker']
        if not speaker_profile:
            return "This profile has no voice registered. (Celebrity)"
        eagle = pveagle.create_recognizer(access_key=access_key, speaker_profiles=[speaker_profile])
        recognizer_recorder = PvRecorder(device_index=-1, frame_length=eagle.frame_length)
        recognizer_recorder.start()
        sum = 0
        for i in range(50):
            audio_frame = recognizer_recorder.read()
            sum += eagle.process(audio_frame)[0]
        recognizer_recorder.stop()
        eagle.delete()
        recognizer_recorder.delete()

        score = sum/50
        if score >= 0.6:
            return "Access granted. Have a great day!"
        else:
            return "Voice does not match. Try again."


def register_user(name, image, voice):
    if not name:
        return "Name field empty."

    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    embeddings = face_recognition.face_encodings(img_rgb)

    # Anti-Spoofing
    label = test(
        image=img_rgb,
        model_dir='./Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
        device_id=0
    )

    if len(embeddings) == 0:
        return "No face detected in the image."
    elif name in user_embeddings:
        return "Name already taken."
    elif label != 1:
        return "Face is fake. Registration denied."

    embedding = embeddings[0]

    eagle_profiler = pveagle.create_profiler(access_key=access_key)
    eagle_profiler.enroll(voice)
    # recorder = PvRecorder(device_index=-1, frame_length=eagle_profiler.min_enroll_samples)
    # recorder.start()
    # enroll_percentage = 0.0
    # while enroll_percentage < 100.0:
    #     audio_frame = recorder.read()
    #     enroll_percentage, feedback = eagle_profiler.enroll(audio_frame)
    #     print(enroll_percentage)
    # recorder.stop()
    speaker_profile = eagle_profiler.export()
    document = {
        'username': name,
        'embedding': embedding.tolist(),  # Convert to list for MongoDB storage,
        'voice_profile': speaker_profile.to_bytes()
    }
    collection.insert_one(document)
    user_embeddings[name] = {}  # Add new data entry to memory
    user_embeddings[name]['face'] = embedding
    user_embeddings[name]['speaker'] = speaker_profile.to_bytes()

    eagle_profiler.delete()
    # recorder.delete()
    return f"User {name} registered successfully!"


def delete_user(name):
    if not name:
        return "Name field empty."

    result = collection.delete_one({'username': name})
    if result.deleted_count > 0:
        user_embeddings.pop(name)  # Remove data entry from memory
        return f'User {name} deleted successfully!'
    else:
        return f'User {name} not found!'


def recognize(img):
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    match = False
    threshold = 0.6
    mini = threshold
    min_dis_id = 'unknown_person'
    for username, embedding in user_embeddings.items():
        face_embedding = embedding['face']
        distance = face_recognition.face_distance([face_embedding], embeddings_unknown)[0]
        if distance < mini:
            mini = distance
            min_dis_id = username
            match = True
    return min_dis_id if match else 'unknown_person'

# Frontend interface
theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont('Roboto'), gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'sans-serif']
)

iface_recognize = gr.Interface(
    fn=recognize_user,
    inputs=[gr.Image(source="webcam", streaming=True)],
    outputs=[gr.HTML()],
    live=True,
    every=1,
    title="Face Recognition Attendance System",
    allow_flagging='never',
    clear_btn=None
)

iface_register = gr.Interface(
    fn=register_user,
    inputs=[gr.Textbox(label="Enter new user name."),
            gr.Image(label="Ensure your face is properly shown.", source="webcam", streaming=True),
            gr.Audio(label="Record your voice.", source="microphone", format="wav")],
    outputs=[gr.HTML()],
    title="Register New User",
    live=False,
    allow_flagging='never',
    clear_btn=None
)

iface_delete = gr.Interface(
    fn=delete_user,
    inputs=[gr.Textbox(label="Enter user name to delete")],
    outputs=[gr.HTML()],
    title="Delete User",
    live=False,
    allow_flagging='never',
    clear_btn=None
)

custom_css = """
    footer {display: none !important;}
    label.float {display: none !important;}
    div.stretch button.secondary {display: none !important;}
    .panel .pending {opacity: 1 !important;}
"""

iface = gr.TabbedInterface([iface_recognize, iface_register, iface_delete], ["Recognize User", "Register User", "Delete User"],
                           css=custom_css, theme=theme)

if __name__ == "__main__":
    iface_recognize.dependencies[0]["show_progress"] = False
    iface_register.dependencies[0]["show_progress"] = False
    iface_delete.dependencies[0]["show_progress"] = False
    iface.launch()
