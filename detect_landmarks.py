import cv2
import dlib
import os

# Define the path to the pre-trained Dlib model
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Initialize Dlib's face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

# Define the path to your dataset folder. It must match your folder structure exactly.
input_folder = "pranav dataset/MP photos"
output_folder = "pranav dataset/MP photos_landmarks"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all images in your input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png")):
        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)
        
        if image is None:
            continue
            
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use the face detector to find all faces in the image
        faces = face_detector(gray_image, 1)

        # Loop through each face found
        for face in faces:
            # Predict the 68 landmarks for the current face
            landmarks = landmark_predictor(gray_image, face)
            
            # Draw circles on the image for visualization
            for i in range(0, 68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Save the image with the landmarks drawn
        output_file_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_file_path, image)
        print(f"Processed and saved: {output_file_path}")