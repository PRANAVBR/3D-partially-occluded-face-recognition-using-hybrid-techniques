import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# --- 1. Set up the Classifier and Feature Extractor ---
def setup_models_and_data(features_folder):
    """
    Loads pre-trained models and trains a kNN classifier on your extracted features.
    """
    known_features = []
    known_labels = []

    # Load all extracted features and their labels
    if not os.path.exists(features_folder):
        print(f"Error: Feature folder not found at '{features_folder}'")
        exit()

    for filename in os.listdir(features_folder):
        if filename.endswith(".npy"):
            # The label is the person's name, extracted from the filename
            label = filename.replace(".npy", "")
            
            feature_path = os.path.join(features_folder, filename)
            feature_vector = np.load(feature_path)
            
            known_features.append(feature_vector)
            known_labels.append(label)
    
    if not known_features:
        print("Error: No feature files found. Please run the feature extraction script first.")
        exit()

    # Initialize and train the kNN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(known_features, known_labels)

    # Load the pre-trained ResNet50 model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()

    return knn, model, known_labels

# --- 2. Feature Extraction from a Live Frame ---
def extract_frame_features(frame, model):
    """
    Preprocesses a video frame and extracts features using the ResNet50 model.
    """
    # Convert OpenCV BGR to PIL RGB format
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Define the image preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(pil_image)
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        features = model(img_tensor)

    return features.squeeze().numpy()

# --- 3. Main Real-Time Loop ---
if __name__ == "__main__":
    # Path to your extracted feature files
    features_folder = "pranav dataset/MP photos_features"
    
    # Set up the classifier and feature extraction model
    knn_classifier, resnet_model, known_labels = setup_models_and_data(features_folder)

    # Use OpenCV to access the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        exit()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Load the pre-trained Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Crop and align the face for feature extraction
            face_crop = frame[y:y+h, x:x+w]
            
            # Extract features from the detected face
            live_face_features = extract_frame_features(face_crop, resnet_model)
            
            # Use the classifier to predict the identity
            prediction_label = knn_classifier.predict([live_face_features])[0]
            
            # Display the result on the video feed
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the live video feed
        cv2.imshow('Live Face Recognition', frame)

        # Press 'q' to exit the live feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()