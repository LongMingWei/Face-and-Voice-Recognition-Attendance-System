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
import struct
import wave
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
    time.sleep(1)
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
        duration = 80
        buffer = 20
        ticks = 0
        valid = (duration-buffer)*0.25
        for i in range(duration):
            if i >= buffer:
                audio_frame = recognizer_recorder.read()
                score = eagle.process(audio_frame)[0]
                if score > 0.1:
                    sum += score
                    ticks += 1
                print(score)
        recognizer_recorder.stop()
        eagle.delete()
        recognizer_recorder.delete()

        if ticks == 0:
            total_score = 0
        else:
            total_score = sum/ticks
        print("Score:", total_score)
        print("Ticks:", ticks)
        if total_score >= 0.8 and ticks >= valid:
            return "Access granted. Have a great day!"
        else:
            return "Voice does not match. Try again."


def register_user(name, voice, image):
    if not name:
        return "Name field empty."
    elif name in user_embeddings:
        return "Name already taken."
    if not voice:
        return "Voice profile missing."

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
    elif label != 1:
        return "Face is fake. Registration denied."

    embedding = embeddings[0]

    eagle_profiler = pveagle.create_profiler(access_key=access_key)
    audio = read_file(voice, eagle_profiler.sample_rate)
    enroll_percentage, feedback = eagle_profiler.enroll(audio)
    if enroll_percentage < 100.0:
        return "Unable to register voice profile. Speak for a longer period of time."
    else:
        speaker_profile = eagle_profiler.export()

    document = {
        'username': name,
        'embedding': embedding.tolist(),  # Convert to list for MongoDB storage,
        'voice_profile': speaker_profile.to_bytes()
    }
    collection.insert_one(document)
    user_embeddings[name] = {}  # Add new data entry to memory
    user_embeddings[name]['face'] = embedding
    user_embeddings[name]['speaker'] = speaker_profile

    eagle_profiler.delete()

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

def read_file(file_name, sample_rate):
    rate_match = True

    with wave.open(file_name, mode="rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()

        params = wav_file.getparams()
        params = list(params)

        if wav_file.getframerate() != sample_rate:
            params[3] = sample_rate
            print(params)
            print("Audio file should have a sample rate of %d. got %d" % (sample_rate, wav_file.getframerate()))
        if sample_width != 2:
            raise ValueError("Audio file should be 16-bit. got %d" % sample_width)
        if channels == 2:
            print("Eagle processes single-channel audio but stereo file is provided. Processing left channel only.")

    if not rate_match:
        with wave.open(file_name, mode="wb") as wav_file:
            wav_file.setparams(params)

    with wave.open(file_name, mode="rb") as wav_file:
        samples = wav_file.readframes(num_frames)

    frames = struct.unpack('h' * num_frames * channels, samples)

    return frames[::channels]

# Frontend interface
theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont('Roboto'), gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'sans-serif']
)

iface_recognize = gr.Interface(
    fn=recognize_user,
    inputs=[gr.Image(label="Show your whole face to the camera", source="webcam", streaming=True)],
    outputs=[gr.HTML()],
    live=True,
    title="Face Recognition Attendance System",
    allow_flagging='never',
    clear_btn=None
)

iface_register = gr.Interface(
    fn=register_user,
    inputs=[gr.Textbox(label="Enter new user name"),
            gr.Audio(label="Record your voice", source="microphone", format="wav", type="filepath"),
            gr.Image(label="Ensure your face is properly shown", source="webcam", streaming=True)],
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
    div.stretch button.secondary {display: none !important;}
    .panel .pending {opacity: 1 !important;}
    .unequal-height {display: block !important;}
    .image-container {width: 50% !important; height: 50% !important;}
    .prose {font-size: 3rem !important;}
"""

iface = gr.TabbedInterface([iface_recognize, iface_register, iface_delete], ["Recognize User", "Register User", "Delete User"],
                           css=custom_css, theme=theme)

if __name__ == "__main__":
    iface_recognize.dependencies[0]["show_progress"] = False
    iface_register.dependencies[0]["show_progress"] = False
    iface_delete.dependencies[0]["show_progress"] = False
    iface.launch()
