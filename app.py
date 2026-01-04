


import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Cancer Detection", layout="centered")

st.title("🧬 Lung & Colon Cancer Detection")
st.write("Upload a histopathology image to get prediction")

# ---------------------------------
# DEVICE
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------
# FEATURE EXTRACTOR (ResNet50)
# ---------------------------------
@st.cache_resource
def load_feature_extractor():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

feature_extractor = load_feature_extractor()

# ---------------------------------
# GAT CLASSIFIER (PLACEHOLDER)
# ---------------------------------
class GATModel(nn.Module):
    def __init__(self, input_dim=2048, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

@st.cache_resource
def load_gat_model():
    model = GATModel()
    model.load_state_dict(torch.load("gat_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

gat_model = load_gat_model()

# ---------------------------------
# CLASS NAMES (EDIT IF NEEDED)
# ---------------------------------
class_names = [
    "Colon Adenocarcinoma",
    "Colon Benign",
    "Lung Adenocarcinoma",
    "Lung Benign",
    "Lung Squamous Cell Carcinoma"
]

# ---------------------------------
# IMAGE PREPROCESSING
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# IMAGE UPLOAD
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = feature_extractor(img_tensor)
            output = gat_model(features)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item() * 100

        st.success(f"Prediction: {class_names[prediction]}")
        st.info(f"Confidence: {confidence:.2f}%")
