import os
import numpy as np
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
from deepface import DeepFace
import util
import sys
import threading

sys.path.append("Silent-Face-Anti-Spoofing")
from test1 import test


class App:
    def __init__(self):
        self.init_main_window()
        self.add_webcam(self.webcam_label)
        self.db_dir = './db'
        self.db = self.load_db()
        self.log_path = './log.txt'
        self.successes = 0
        self.login()

    def init_main_window(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("720x650+350+100")

        self.webcam_label = tk.Label(self.main_window, bg='black')
        self.webcam_label.place(x=10, y=10, width=700, height=500)
        self.webcam_label.configure(highlightbackground="black", highlightthickness=5)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'Register new user', 'gray', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=180, y=550)

        self.recognized_face_label = tk.Label(self.main_window, text="", font=("Verdana", 16))
        self.recognized_face_label.place(x=10, y=510, width=700, height=50)

    def load_db(self):
        db = {}
        filenames = os.listdir(self.db_dir)
        for filename in filenames:
            path_ = os.path.join(self.db_dir, filename)
            with open(path_, 'rb') as file:
                embeddings = pickle.load(file)
                db[filename[:-7]] = embeddings
                print(filename[:-7])
        print('loading db: done')
        return db

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (700, 500))  # Resize the frame to fit the label
            self.most_recent_capture_arr = frame
            self.img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(self.img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
        self._label.after(10, self.process_webcam)

    def login(self):
        # Run login in a separate thread
        threading.Thread(target=self.login_thread).start()

    def login_thread(self):
        label = test(
            image=self.most_recent_capture_arr,
            model_dir='./Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
            device_id=0
        )

        if label == 1:
            if len(face_recognition.face_encodings(self.img_)) == 0:
                self.update_ui("No person detected", "black")
            else:
                name = util.recognize(self.most_recent_capture_arr, self.db)
                self.handle_recognition(name)
        else:
            self.handle_fake_recognition()

        self.main_window.after(1500, self.login)

    def handle_recognition(self, name):
        if name in ['unknown_person', 'no_persons_found']:
            self.update_ui("Unknown user. Please register if you have not or try again.", "red")
        else:
            result = DeepFace.analyze(np.array(self.most_recent_capture_arr), actions=['emotion'], enforce_detection=False)
            mood = result[0]['dominant_emotion']
            self.update_ui(f"Access granted. Welcome, {name}, you look {mood} today!", "green")

            with open(self.log_path, 'a') as f:
                f.write(f'{name},{datetime.datetime.now()},in\n')

    def handle_fake_recognition(self):
        name = util.recognize(self.most_recent_capture_arr, self.db)
        if name in ['unknown_person', 'no_persons_found']:
            self.update_ui("Hello unknown fake NPC. You shall not pass!", "red")
        else:
            self.update_ui(f"You think you are {name}? You shall not pass!", "red")

        with open(self.log_path, 'a') as f:
            f.write(f'{name},{datetime.datetime.now()},in\n')

    def update_ui(self, text, color):
        self.main_window.after(0, self._update_ui, text, color)

    def _update_ui(self, text, color):
        self.recognized_face_label.config(text=text)
        self.webcam_label.configure(highlightbackground=color)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x600+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user
        )
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Cancel', 'red', self.try_again_register_new_user
        )
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = tk.Label(self.register_new_user_window, text="Input username:", font=("Verdana", 20))
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

        with open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb') as file:
            pickle.dump(embeddings, file)

        self.db[name] = embeddings
        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
