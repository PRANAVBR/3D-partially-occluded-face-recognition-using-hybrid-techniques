import cv2
import dlib
import numpy as np
import os

# Define the path to the pre-trained Dlib model
predictor_path = "shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(predictor_path)

# Initialize Dlib's face detector
face_detector = dlib.get_frontal_face_detector()

# Define the paths for your input and output folders
input_folder = "pranav dataset/MP photos"
aligned_folder = "pranav dataset/MP photos_aligned"

if not os.path.exists(aligned_folder):
    os.makedirs(aligned_folder)

# Define the desired size for the aligned faces
desired_face_width = 256
desired_face_height = 256
desired_left_eye = (0.35, 0.35)

# Function to align a single face
def align_face(image, face_rect):
    shape = landmark_predictor(image, face_rect)
    
    left_eye = shape.part(36)
    right_eye = shape.part(45)

    left_eye_center = np.array([left_eye.x, left_eye.y])
    right_eye_center = np.array([right_eye.x, right_eye.y])

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    dist_between_eyes = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist_between_eyes = desired_face_width * (1 - 2 * desired_left_eye[0])
    scale = desired_dist_between_eyes / dist_between_eyes

    eyes_center = ((left_eye_center + right_eye_center) / 2).astype("int")
    
    # Corrected line: Explicitly pass a tuple of integers
    M = cv2.getRotationMatrix2D(tuple(eyes_center.astype(float)), angle, scale)

    tX = desired_face_width * 0.5 - eyes_center[0]
    tY = desired_face_height * desired_left_eye[1] - eyes_center[1]
    M[0, 2] += tX
    M[1, 2] += tY

    aligned_face = cv2.warpAffine(image, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)
    
    return aligned_face

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png")):
        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)

        if image is None:
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector(gray_image, 1)

        for face in faces:
            aligned_face = align_face(gray_image, face)
            
            output_path = os.path.join(aligned_folder, filename)
            cv2.imwrite(output_path, aligned_face)
            print(f"Processed and aligned: {output_path}")