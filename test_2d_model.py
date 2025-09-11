import os
import numpy as np
import dlib
import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

# --- Part 1: Feature Extraction from a New Image ---
def extract_new_image_features(image_path, model):
    """
    Loads an image, preprocesses it, and extracts features using the ResNet50 model.
    """
    # Define the image preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension
    
    with torch.no_grad():
        features = model(img_tensor)

    return features.squeeze().numpy()

# --- Part 2: Train and Test the Classifier ---
def train_and_test_classifier(features_folder, test_image_path):
    """
    Trains a kNN classifier and tests it on a new image.
    """
    known_features = []
    known_labels = []

    # Load all extracted features and their labels
    for filename in os.listdir(features_folder):
        if filename.endswith(".npy"):
            # The label is the person's name, extracted from the filename
            label = filename.split('_')[0]
            
            feature_path = os.path.join(features_folder, filename)
            feature_vector = np.load(feature_path)
            
            known_features.append(feature_vector)
            known_labels.append(label)

    # Initialize and train the kNN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(known_features, known_labels)

    # Load the pre-trained ResNet50 model for feature extraction
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Extract features from the new test image
    test_feature_vector = extract_new_image_features(test_image_path, model)

    # Make a prediction on the new image
    prediction = knn.predict([test_feature_vector])

    return prediction[0]

# --- Main Execution ---
if __name__ == "__main__":
    # You need to have a folder with your extracted .npy files
    features_folder = "pranav dataset/MP photos_features"
    
    # Create a new image for testing. For example, copy one of your original images.
    test_image_path = "pranav dataset/MP photos_aligned/front1 with specs.jpg"

    print(f"Testing the model with a new image: {test_image_path}")
    
    # Make sure you have a correctly trained model or run the training here
    # Since we are training a new one, we don't need a model checkpoint file
    predicted_label = train_and_test_classifier(features_folder, test_image_path)
    
    print(f"The model predicted the person is: {predicted_label}")