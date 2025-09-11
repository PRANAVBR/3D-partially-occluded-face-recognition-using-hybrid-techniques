import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np  # <--- This is the line you need to add

# 1. Load the pre-trained ResNet50 model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# 2. Remove the final classification layer
# The original model's job is to classify 1000 different objects.
# We remove the final layer ('fc') to get the raw features, not a classification.
model = nn.Sequential(*list(model.children())[:-1])
model.eval()  # Set the model to evaluation mode

# 3. Define the image preprocessing steps
# Images must be transformed to match the format the model was trained on.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the paths for your aligned images and the output features
aligned_folder = "pranav dataset/MP photos_aligned"
output_features_folder = "pranav dataset/MP photos_features"

if not os.path.exists(output_features_folder):
    os.makedirs(output_features_folder)

# Loop through all aligned images
for filename in os.listdir(aligned_folder):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join(aligned_folder, filename)
        
        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension
        
        # Extract features using the modified model
        with torch.no_grad():
            features = model(img_tensor)

        # The features will be a tensor of shape [1, 2048, 1, 1].
        # We need to flatten it into a 1D vector.
        feature_vector = features.squeeze().numpy()
        
        # Save the feature vector as a .npy file
        output_path = os.path.join(output_features_folder, filename.split('.')[0] + '.npy')
        np.save(output_path, feature_vector)
        
        print(f"Extracted and saved features for: {filename}")