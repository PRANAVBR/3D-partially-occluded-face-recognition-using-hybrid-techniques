import dlib
import cv2
import os
import numpy as np

def get_5_landmarks(landmarks):
    # This function extracts the 5 key landmarks needed by the 3D reconstruction model
    # ... (rest of the function is the same)
    left_eye_x = np.mean([landmarks.part(i).x for i in range(36, 42)])
    left_eye_y = np.mean([landmarks.part(i).y for i in range(36, 42)])

    right_eye_x = np.mean([landmarks.part(i).x for i in range(42, 48)])
    right_eye_y = np.mean([landmarks.part(i).y for i in range(42, 48)])

    nose_x = landmarks.part(30).x
    nose_y = landmarks.part(30).y

    left_mouth_x = landmarks.part(48).x
    left_mouth_y = landmarks.part(48).y

    right_mouth_x = landmarks.part(54).x
    right_mouth_y = landmarks.part(54).y

    return np.array([
        [left_eye_x, left_eye_y],
        [right_eye_x, right_eye_y],
        [nose_x, nose_y],
        [left_mouth_x, left_mouth_y],
        [right_mouth_x, right_mouth_y]
    ])

if __name__ == "__main__":
    # Define the path to the Dlib landmark model
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    landmark_predictor = dlib.shape_predictor(predictor_path)
    face_detector = dlib.get_frontal_face_detector()

    input_folder = "my_aligned_photos"
    output_folder = os.path.join(input_folder, "detections")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png")):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)

            if image is None:
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray_image, 1)

            if len(faces) == 0:
                print(f"No face detected in {filename}. Skipping.")
                continue

            for face in faces:
                landmarks = landmark_predictor(gray_image, face)
                five_landmarks = get_5_landmarks(landmarks)

                output_path = os.path.join(output_folder, filename.split('.')[0] + '.txt')
                np.savetxt(output_path, five_landmarks)
                print(f"Created detection file for: {filename}")