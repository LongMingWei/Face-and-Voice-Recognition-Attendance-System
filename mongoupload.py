import pymongo
import pickle
import os
import numpy as np

# Replace with your actual MongoDB URI
mongo_uri = "hidden"

# Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)
db = client['users']
collection = db['faces']

# Directory containing the pickle files
pickle_directory = './db'  # Adjust the path as needed

# Ensure the database directory exists
if not os.path.exists(pickle_directory):
    os.makedirs(pickle_directory)

# Function to read pickle file and return the content
def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Loop through all pickle files in the directory
for filename in os.listdir(pickle_directory):
    if filename.endswith('.pickle'):
        file_path = os.path.join(pickle_directory, filename)
        username = os.path.splitext(filename)[0]  # Extract username without extension
        pickle_data = read_pickle_file(file_path)
        if isinstance(pickle_data, np.ndarray):
            pickle_data = pickle_data.tolist()  # Convert numpy array to list
        document = {
            'username': username,
            'embedding': pickle_data
        }
        collection.insert_one(document)
        print(f"Inserted data for {username}")

print("All data inserted successfully.")
