import os
import pickle

import tkinter as tk
from tkinter import messagebox
import face_recognition


def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Verdana', 20)
                    )

    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Verdana", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)




##    def process_images(input_dir, output_dir):
##    # Ensure the output directory exists
##        if not os.path.exists(output_dir):
##            os.makedirs(output_dir)
##
##        # Walk through the directory
##        for root, dirs, files in os.walk(input_dir):
##            for file in files:
##                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
##                    if int(file.lower()[-8:-4]) >= 4:
##                        continue
##                        
##                    # Get the relative path of the file
##                    rel_dir = os.path.relpath(root, input_dir)
##                    rel_file = os.path.join(rel_dir, file)
##                    
##                    # Ensure the corresponding output subdirectory exists
##                    output_subdir = os.path.join(output_dir, rel_dir)
##                    if not os.path.exists(output_subdir):
##                        os.makedirs(output_subdir)
##                    
##                    # Open and process the image file
##                    img_path = os.path.join(root, file)
##                    with Image.open(img_path) as img:
##                        img_array = np.array(img)
##                        embeddings = face_recognition.face_encodings(img_array)
##                        if len(embeddings) >= 1:
##                            # Serialize the image array to a pickle file
##                            pickle_file = os.path.splitext(file)[0] + '.pickle'
##                            pickle_path = os.path.join(output_subdir, pickle_file)
##                            with open(pickle_path, 'wb') as pkl_file:
##                                pickle.dump(embeddings[0], pkl_file)
##                                                            
##                            print(f"Processed {img_path} -> {pickle_path}")

##    def move_files_to_single_folder(input_dir, output_dir):
##        # Ensure the output directory exists
##        if not os.path.exists(output_dir):
##            os.makedirs(output_dir)
##
##        # Walk through the directory
##        for root, dirs, files in os.walk(input_dir):
##            for file in files:
##                # Construct the full file path
##                file_path = os.path.join(root, file)
##                
##                # Construct the destination path
##                dest_path = os.path.join(output_dir, file)
##                
##                # Ensure no name collision by appending a unique number if needed
##                base, extension = os.path.splitext(file)
##                counter = 1
##                while os.path.exists(dest_path):
##                    dest_path = os.path.join(output_dir, f"{base}_{counter}{extension}")
##                    counter += 1
##                
##                # Move the file using os.rename
##                os.rename(file_path, dest_path)
##                print(f"Moved {file_path} -> {dest_path}")
##
####    # Example usage:
##    input_directory = './db'  ##'./archive/lfw-deepfunneled/lfw-deepfunneled'
##    output_directory = './db'
##    move_files_to_single_folder(input_directory, output_directory)

