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
from scipy.signal import resample
import struct
import wave
import sys
import os
import time

sys.path.append("Silent-Face-Anti-Spoofing")
from test1 import test  # from test1.py in the Silent-Face-Anti-Spoofing folder
# gr.themes.builder()

uri = os.getenv('MONGODBPASSCODE')
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

access_key = os.getenv('PVRECORDERACCESSKEY')


state = 1
usersname=""

def recognize_user(image, voice):
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
                result = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)
                mood = result[0]['dominant_emotion']
                usersname = name
                if name.endswith('_0001') or name.endswith('_0002') or name.endswith('_0003'):
                    name = name[:-5]
                verification = f"Hello {name}, you look {mood} today! "
                speaker_profile = user_embeddings[usersname]['speaker']
                if not speaker_profile:
                    verification += "This profile has no voice registered. (Celebrity)"
                else:
                    verification += "Verify your voice: Press 'Record from microphone' and SPEAK now."
                    state = 2
                return verification
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
        if not voice:
            return "No voice input provided. Please try again."
            
        eagle = pveagle.create_recognizer(access_key=access_key, speaker_profiles=[speaker_profile])

        mismatch = False
        with wave.open(voice, mode="rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            num_frames = wav_file.getnframes()
            input_sample_rate = wav_file.getframerate()

            if sample_width != 2:
                raise ValueError(f"Incorrect sample width: expected 16-bit, got {sample_width * 8}-bit") 
            if input_sample_rate != eagle.sample_rate:
                mismatch = True
                num_frames = int(num_frames * eagle.sample_rate / input_sample_rate)

        if mismatch:
            with wave.open(voice, mode="rb") as wav_file:
                frames = wav_file.readframes(num_frames)
                audio_data = struct.unpack('h' * num_frames * channels, frames)
                audio_data = np.array(audio_data[::channels], dtype=np.float32)  
                audio_data = resample(audio_data, num_frames)  
        else:
            with wave.open(voice, mode="rb") as wav_file:
                frames = wav_file.readframes(num_frames)
                audio_data = struct.unpack('h' * num_frames * channels, frames)
                audio_data = np.array(audio_data[::channels], dtype=np.float32)  

        sum_scores = 0
        duration = len(audio_data) // eagle.frame_length
        buffer = 100
        ticks = 0
        valid = (duration - buffer) * 0.25

        for i in range(buffer, duration):
            start_idx = i * eagle.frame_length
            end_idx = (i + 1) * eagle.frame_length
            frame = audio_data[start_idx:end_idx].astype(np.int32)
            if len(frame) < eagle.frame_length:
                break  
            score = eagle.process(frame)[0]
            if score > 0.2:
                sum_scores += score
                ticks += 1
            print(score)
        
        eagle.delete()        

        if ticks == 0:
            total_score = 0
        else:
            total_score = sum_scores/ticks
        print("Score:", total_score)
        print("Ticks:", ticks)
        if total_score >= 0.6 and ticks >= valid:
            return "Access granted. Have a great day! (Press 'Stop recording')"
        else:
            return "Voice does not match. Try again. (Press 'Stop recording')"


def register_user(image, name, voice):
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
    inputs=[gr.Image(label="Show your whole face to the camera", source="webcam", streaming=True),
            gr.Audio(show_label=False, source="microphone", streaming=True, format="wav", type="filepath")],
    outputs=[gr.HTML()],
    live=True,
    title="Face Recognition Attendance System",
    allow_flagging='never',
    clear_btn=None
)

iface_register = gr.Interface(
    fn=register_user,
    inputs=[gr.Image(label="Ensure your face is properly shown and details are entered below", source="webcam", streaming=True),
            gr.Textbox(label="Enter new user name"),
            gr.Audio(label="Record your voice (15-20 seconds recommended)", source="microphone", format="wav", type="filepath", show_edit_button=False, show_download_button=False)],
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
    .tab-nav button {font-size: 1.5rem !important;}
    .prose {font-size: 3rem !important;}
    .gap.panel {border: none !important;}
"""

iface = gr.TabbedInterface([iface_register, iface_recognize, iface_delete], ["Register User", "Verify User", "Delete User"],
                           css=custom_css, theme=theme)

if __name__ == "__main__":
    iface_recognize.dependencies[0]["show_progress"] = False
    iface_register.dependencies[0]["show_progress"] = False
    iface_delete.dependencies[0]["show_progress"] = False
    iface.launch()
