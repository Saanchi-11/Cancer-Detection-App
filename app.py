"""
Simplified Streamlit App - Uses YOUR trained model
This version has better error handling and diagnostics
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Cancer Detection", page_icon="🔬", layout="wide")

# Title
st.title("🔬 Cancer Detection System")
st.markdown("---")

# Step 1: Check what files exist
st.sidebar.header("📁 File Check")
files_to_check = ['gat_model.pth', 'metadata.pkl', 'graph_data.pt']
files_exist = {}

for f in files_to_check:
    exists = os.path.exists(f)
    files_exist[f] = exists
    if exists:
        size = os.path.getsize(f) / (1024*1024)
        st.sidebar.success(f"✅ {f} ({size:.1f} MB)")
    else:
        st.sidebar.error(f"❌ {f} MISSING")

# Check for features file (might have different names)
features_file = None
for fname in ['features.npz', 'feature_embeddings.npz', 'X.npz']:
    if os.path.exists(fname):
        features_file = fname
        size = os.path.getsize(fname) / (1024*1024)
        st.sidebar.success(f"✅ {fname} ({size:.1f} MB)")
        break

if not features_file:
    st.sidebar.error("❌ No features file found")

st.sidebar.markdown("---")

# Try to import torch_geometric
try:
    from torch_geometric.nn import GATConv
    st.sidebar.success("✅ torch_geometric imported")
except ImportError as e:
    st.sidebar.error(f"❌ torch_geometric: {e}")
    st.error("Installing torch-geometric...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "torch-geometric", "torch-scatter", "torch-sparse", "torch-cluster",
                          "--extra-index-url", "https://data.pyg.org/whl/torch-2.1.0+cpu.html"])
    st.success("Installed! Please refresh page.")
    st.stop()

# Import other required packages
try:
    import timm
    from torchvision import transforms
    from sklearn.metrics.pairwise import cosine_similarity
    import pickle
    st.sidebar.success("✅ All packages imported")
except ImportError as e:
    st.sidebar.error(f"❌ Missing package: {e}")
    st.stop()

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

# Load functions with detailed error messages
@st.cache_resource
def load_all_components():
    """Load everything with detailed diagnostics"""
    
    components = {
        'model': None,
        'X': None,
        'edge_index': None,
        'metadata': None,
        'vit': None,
        'errors': []
    }
    
    device = torch.device('cpu')  # Use CPU for Streamlit Cloud
    
    # 1. Load metadata
    try:
        with open('metadata.pkl', 'rb') as f:
            components['metadata'] = pickle.load(f)
        st.sidebar.info(f"Classes: {components['metadata']['class_names']}")
    except Exception as e:
        components['errors'].append(f"Metadata error: {str(e)}")
        return components
    
    # 2. Load model checkpoint
    try:
        checkpoint = torch.load('gat_model.pth', map_location=device)
        
        # Check checkpoint structure
        if 'model_state_dict' in checkpoint:
            # It's a full checkpoint
            config = checkpoint['model_config']
            model = GAT(**config)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.sidebar.info(f"Model config: {config}")
        else:
            # It's just state dict
            st.warning("Checkpoint is just state_dict, trying to load...")
            # Try to infer dimensions
            metadata = components['metadata']
            model = GAT(in_dim=768, hidden_dim=128, out_dim=len(metadata['class_names']))
            model.load_state_dict(checkpoint)
        
        model.eval()
        components['model'] = model
        
    except Exception as e:
        components['errors'].append(f"Model error: {str(e)}")
        return components
    
    # 3. Load features
    try:
        if features_file:
            features = np.load(features_file)
            # Try different possible key names
            for key in ['X', 'X_all', 'X_train']:
                if key in features:
                    components['X'] = features[key]
                    st.sidebar.info(f"Features shape: {components['X'].shape}")
                    break
            
            if components['X'] is None:
                components['errors'].append(f"No valid key in features file. Keys: {list(features.keys())}")
                return components
        else:
            components['errors'].append("Features file not found")
            return components
            
    except Exception as e:
        components['errors'].append(f"Features error: {str(e)}")
        return components
    
    # 4. Load graph
    try:
        graph = torch.load('graph_data.pt', map_location=device)
        components['edge_index'] = graph['edge_index']
        st.sidebar.info(f"Graph edges: {components['edge_index'].shape[1]}")
    except Exception as e:
        components['errors'].append(f"Graph error: {str(e)}")
        return components
    
    # 5. Load ViT
    try:
        vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        vit.head = nn.Identity()
        vit.eval()
        components['vit'] = vit
    except Exception as e:
        components['errors'].append(f"ViT error: {str(e)}")
        return components
    
    return components

# Load everything
with st.spinner("Loading model..."):
    components = load_all_components()

# Check for errors
if components['errors']:
    st.error("❌ Failed to load components:")
    for err in components['errors']:
        st.error(err)
    
    st.info("**Debug Info:**")
    st.json({
        "model_loaded": components['model'] is not None,
        "features_loaded": components['X'] is not None,
        "graph_loaded": components['edge_index'] is not None,
        "metadata_loaded": components['metadata'] is not None,
        "vit_loaded": components['vit'] is not None
    })
    st.stop()

st.sidebar.success("✅ All components loaded!")

# Prediction function
def predict_cancer(image, components):
    """Predict cancer type"""
    
    # Preprocess
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_resized = image.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = transforms.Normalize([0.5]*3, [0.5]*3)(img_tensor)
    
    # Extract features
    with torch.no_grad():
        features = components['vit'](img_tensor).numpy()
    
    # k-NN
    KNN_K = components['metadata'].get('knn_k', 10)
    sims = cosine_similarity(features, components['X'])[0]
    neighbors = sims.argsort()[-KNN_K:]
    
    # Build graph
    new_node = len(components['X'])
    old_edges = components['edge_index'].numpy()
    new_edges = np.vstack([np.repeat(new_node, KNN_K), neighbors])
    all_edges = np.hstack([old_edges, new_edges, new_edges[::-1]])
    graph_edges = torch.LongTensor(all_edges)
    
    # Predict
    all_features = torch.FloatTensor(np.vstack([components['X'], features]))
    
    with torch.no_grad():
        output = components['model'](all_features, graph_edges)
        probs = torch.softmax(output[new_node], dim=0).numpy()
    
    pred_idx = probs.argmax()
    pred_class = components['metadata']['class_names'][pred_idx]
    
    return pred_class, probs[pred_idx], probs, img_resized

# UI
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    uploaded = st.file_uploader("Choose histopathology image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption=f"Original ({image.size[0]}×{image.size[1]})", use_container_width=True)

with col2:
    st.header("📊 Prediction")
    
    if uploaded:
        with st.spinner("Analyzing..."):
            try:
                pred_class, confidence, probs, img_resized = predict_cancer(image, components)
                
                st.image(img_resized, caption="Preprocessed (224×224)", width=224)
                
                # Result
                if confidence > 0.90:
                    st.success(f"### {pred_class}")
                    st.success(f"Confidence: {confidence:.2%}")
                elif confidence > 0.70:
                    st.warning(f"### {pred_class}")
                    st.warning(f"Confidence: {confidence:.2%}")
                else:
                    st.error(f"### {pred_class}")
                    st.error(f"Confidence: {confidence:.2%}")
                
                # Probabilities
                st.markdown("**All Classes:**")
                for i, cls in enumerate(components['metadata']['class_names']):
                    st.progress(float(probs[i]), text=f"{cls}: {probs[i]:.2%}")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Upload an image to classify")

st.markdown("---")
st.caption("⚠️ For research and educational purposes only")
