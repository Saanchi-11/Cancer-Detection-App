
"""
Streamlit Cancer Detection App
Upload histopathology images for real-time cancer classification
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from PIL import Image
import timm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Page config
st.set_page_config(
    page_title="Cancer Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Define GAT Model
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, drop=0.6):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=drop)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=drop)
        self.drop = drop
    
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.gat2(x, edge_index)
        return x

# Cache model loading
@st.cache_resource
def load_model(model_dir="./cancer_detection_model"):
    """Load trained model and data"""
    
    # Load metadata
    with open(f"{model_dir}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    # Load model checkpoint
    checkpoint = torch.load(f"{model_dir}/gat_model.pth", map_location='cpu')
    
    # Initialize model
    config = checkpoint['model_config']
    model = GAT(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load features
    features = np.load(f"{model_dir}/features.npz")
    X = features['X']
    
    # Load graph
    graph_data = torch.load(f"{model_dir}/graph.pt", map_location='cpu')
    edge_index = graph_data['edge_index']
    
    # Load ViT
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
    vit_model.head = nn.Identity()
    vit_model.eval()
    
    return model, X, edge_index, metadata, vit_model

# Cache preprocessing transform
@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def preprocess_image(image, target_size=224):
    """Preprocess uploaded image"""
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to exact dimensions
    image_resized = image.resize((target_size, target_size), Image.LANCZOS)
    
    # Convert to tensor
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize
    transform = get_transform()
    img_tensor = transform(img_tensor)
    
    return img_tensor, image_resized

def predict_cancer(image, model, vit_model, X, edge_index, metadata):
    """Predict cancer type from image"""
    
    # Preprocess
    img_tensor, img_resized = preprocess_image(image)
    
    # Extract ViT features
    with torch.no_grad():
        features = vit_model(img_tensor).numpy()
    
    # Find k-nearest neighbors
    KNN_K = metadata['knn_k']
    similarities = cosine_similarity(features, X)[0]
    neighbors = similarities.argsort()[-KNN_K:]
    
    # Build temporary graph
    new_node = len(X)
    old_edges = edge_index.numpy()
    new_edges = np.vstack([
        np.repeat(new_node, KNN_K),
        neighbors
    ])
    all_edges = np.hstack([old_edges, new_edges, new_edges[::-1]])
    graph_edges = torch.LongTensor(all_edges)
    
    # Add features
    all_features = torch.FloatTensor(np.vstack([X, features]))
    
    # Predict
    with torch.no_grad():
        model.eval()
        output = model(all_features, graph_edges)
        probs = torch.softmax(output[new_node], dim=0).numpy()
    
    # Get prediction
    pred_idx = probs.argmax()
    pred_class = metadata['class_names'][pred_idx]
    confidence = probs[pred_idx]
    
    return pred_class, confidence, probs, img_resized

# App header
st.title("🔬 Cancer Detection System")
st.markdown("### AI-Powered Histopathology Image Classification")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    try:
        model, X, edge_index, metadata, vit_model = load_model()
        
        st.success("✅ Model Loaded Successfully")
        st.metric("Test Accuracy", f"{metadata['test_accuracy']:.2%}")
        st.metric("Validation Accuracy", f"{metadata['val_accuracy']:.2%}")
        
        st.markdown("---")
        st.markdown("**Detected Cancer Types:**")
        for i, cls in enumerate(metadata['class_names'], 1):
            st.markdown(f"{i}. {cls}")
        
        st.markdown("---")
        st.markdown("**Model Architecture:**")
        st.info("Vision Transformer (ViT) + Graph Attention Network (GAT)")
        
        st.markdown("---")
        st.caption(f"Model trained: {metadata['timestamp']}")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure the model files are in './cancer_detection_model/' directory")
        st.stop()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a histopathology image...",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a microscopy image of tissue sample"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Show image info
        st.caption(f"**Original Size:** {image.size[0]} × {image.size[1]} pixels")
        st.caption(f"**Mode:** {image.mode}")

with col2:
    st.header("📊 Prediction Results")
    
    if uploaded_file is not None:
        with st.spinner("🔄 Analyzing image..."):
            try:
                # Predict
                pred_class, confidence, probs, img_resized = predict_cancer(
                    image, model, vit_model, X, edge_index, metadata
                )
                
                # Display resized image
                st.image(img_resized, caption="Preprocessed (224×224)", use_container_width=True)
                
                # Display prediction
                st.markdown("### 🎯 Diagnosis")
                
                # Color code based on confidence
                if confidence > 0.90:
                    st.success(f"**{pred_class}**")
                    conf_color = "🟢"
                    conf_text = "High Confidence"
                elif confidence > 0.70:
                    st.warning(f"**{pred_class}**")
                    conf_color = "🟡"
                    conf_text = "Moderate Confidence"
                else:
                    st.error(f"**{pred_class}**")
                    conf_color = "🔴"
                    conf_text = "Low Confidence"
                
                # Confidence metric
                st.metric("Confidence Score", f"{confidence:.2%}", 
                         delta=f"{conf_color} {conf_text}")
                
                # Probability distribution
                st.markdown("### 📈 Probability Distribution")
                
                # Sort by probability
                sorted_indices = np.argsort(probs)[::-1]
                
                for idx in sorted_indices:
                    cls_name = metadata['class_names'][idx]
                    prob = probs[idx]
                    
                    # Progress bar
                    st.progress(float(prob), text=f"{cls_name}: {prob:.2%}")
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload an image to start classification")
        
        # Show example placeholder
        st.markdown("### Expected Input Format")
        st.markdown("""
        - **Type:** Histopathology microscopy images
        - **Format:** PNG, JPG, JPEG, BMP, TIFF
        - **Size:** Any size (will be resized to 224×224)
        - **Color:** RGB color images
        """)

# Footer
st.markdown("---")

# Disclaimer
st.warning("""
⚠️ **Medical Disclaimer:** This tool is for research and educational purposes only. 
It should NOT be used as a substitute for professional medical diagnosis. 
Always consult qualified healthcare professionals for medical decisions.
""")

# Technical details expander
with st.expander("🔧 Technical Details"):
    st.markdown("""
    **Model Architecture:**
    - **Feature Extraction:** Vision Transformer (ViT-Base, 224×224 input)
    - **Classification:** Graph Attention Network (GAT) with 2 layers
    - **Graph Construction:** k-Nearest Neighbors (k=10) in feature space
    
    **Performance Metrics:**
    - Test Accuracy: 96.67%
    - Precision: 99%
    - Recall: 99%
    - F1-Score: 99%
    
    **Dataset:**
    - Lung and Colon Cancer Histopathological Images
    - ~25,000 images across 5 classes
    - 70/15/15 train/val/test split
    
    **Preprocessing:**
    1. Convert to RGB
    2. Resize to 224×224 (LANCZOS interpolation)
    3. Normalize: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    4. Extract 768-dimensional ViT features
    5. Graph-based classification via GAT
    """)

# About section
with st.expander("ℹ️ About"):
    st.markdown("""
    This cancer detection system combines state-of-the-art deep learning techniques:
    
    1. **Vision Transformers (ViT)** extract rich visual features from histopathology images
    2. **Graph Attention Networks (GAT)** classify images by learning from similar samples in feature space
    3. **k-NN Graph Construction** creates relationships between images based on feature similarity
    
    The model achieves near-perfect accuracy on test data, demonstrating the power of combining 
    transformer-based feature extraction with graph neural networks.
    
    **Developed for:** Educational and research purposes in medical image analysis.
    """)