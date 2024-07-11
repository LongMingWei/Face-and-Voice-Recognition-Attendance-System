from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import cv2
import numpy as np
import face_recognition
from PIL import Image
import gradio as gr
from deepface import DeepFace
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
        username = user['username']
        embedding = np.array(user['embedding'])
        user_embeddings[username] = embedding
        print(username)

load_embeddings()

def recognize_user(image):
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Anti-Spoofing
    label = test(
        image=img_rgb,
        model_dir='./Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
        device_id=0
    )

    if label == 1:
        if len(face_recognition.face_encodings(img_rgb)) == 0:
            return "No person detected"
        else:
            name = recognize(img_rgb)
            if name in ['unknown_person', 'no_persons_found']:
                return "Unknown user. Please register if you have not or try again."
            else:
                result = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)
                mood = result[0]['dominant_emotion']
                if name.endswith('_0001') or name.endswith('_0002') or name.endswith('_0003'):
                    name = name[:-5]
                return f"Access granted. Welcome, {name}, you look {mood} today!"
    else:
        name = recognize(img_rgb)
        if name in ['unknown_person', 'no_persons_found']:
            return "Hello unknown fake NPC. You shall not pass!"
        else:
            if name.endswith('_0001') or name.endswith('_0002') or name.endswith('_0003'):
                name = name[:-5]
            return f"You think you are {name}? You shall not pass!"

def register_user(name, image):
    if not name:
        return "Enter new user name."

    img = np.array(image)
    embeddings = face_recognition.face_encodings(img)
    if len(embeddings) == 0:
        return "No face detected in the image."
    embedding = embeddings[0].tolist()  # Convert to list for MongoDB storage
    document = {
        'username': name,
        'embedding': embedding
    }
    collection.insert_one(document)
    user_embeddings[name] = embeddings[0]  # Add new data entry to memory
    return f'User {name} registered successfully!'

def delete_user(name):
    if not name:
        return "Enter user name to delete."

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
        distance = face_recognition.face_distance([embedding], embeddings_unknown)[0]
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
    title="Face Recognition Attendance System",
    allow_flagging='never',
    clear_btn=None
)

iface_register = gr.Interface(
    fn=register_user,
    inputs=[gr.Textbox(show_label=False), gr.Image(source="webcam", streaming=True)],
    outputs=[gr.HTML()],
    title="Register New User",
    live=False,
    allow_flagging='never',
    clear_btn=None
)

iface_delete = gr.Interface(
    fn=delete_user,
    inputs=[gr.Textbox(show_label=False)],
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
