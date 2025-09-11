import urllib.request
import bz2
import os

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
output_filename = "shape_predictor_68_face_landmarks.dat"

print("Downloading the model file...")
# Download the compressed file
urllib.request.urlretrieve(url, "temp_model.dat.bz2")

print("Download complete. Decompressing the file...")
# Decompress the file
with open("temp_model.dat.bz2", "rb") as source, open(output_filename, "wb") as dest:
    dest.write(bz2.decompress(source.read()))

print(f"File successfully saved as: {output_filename}")

# Clean up the temporary compressed file
os.remove("temp_model.dat.bz2")