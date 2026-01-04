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

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define GAT Model (MUST match training)
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

@st.cache_resource
def load_model():
    """Load model with CORRECT checkpoint structure"""
    
    try:
        # 1. Load checkpoint (it's a dictionary!)
        checkpoint = torch.load("gat_model.pth", map_location=device)
        
        # 2. Get model config from checkpoint
        config = checkpoint['model_config']
        
        # 3. Initialize model with saved config
        model = GAT(
            in_dim=config['in_dim'],
            hidden_dim=config['hidden_dim'],
            out_dim=config['out_dim'],
            heads=config['heads'],
            drop=config['drop']
        )
        
        # 4. Load ONLY the model weights (not the whole checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded | Val Acc: {checkpoint['best_val_acc']:.2%}")
        
        return model, checkpoint
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

@st.cache_resource
def load_features():
    """Load training features"""
    try:
        # Check for different possible filenames
        if os.path.exists("features.npz"):
            features = np.load("features.npz")
        elif os.path.exists("feature_embeddings.npz"):
            features = np.load("feature_embeddings.npz")
        else:
            st.error("❌ Features file not found")
            return None
        
        X = features['X'] if 'X' in features else features['X_all']
        st.success(f"✅ Features loaded: {X.shape}")
        return X
        
    except Exception as e:
        st.error(f"❌ Error loading features: {str(e)}")
        return None

@st.cache_resource
def load_graph():
    """Load graph structure"""
    try:
        graph = torch.load("graph_data.pt", map_location=device)
        edge_index = graph['edge_index']
        st.success(f"✅ Graph loaded: {graph['num_nodes']} nodes")
        return edge_index
        
    except Exception as e:
        st.error(f"❌ Error loading graph: {str(e)}")
        return None

@st.cache_resource
def load_metadata():
    """Load metadata"""
    try:
        with open("metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        st.success(f"✅ Metadata loaded: {len(metadata['class_names'])} classes")
        return metadata
        
    except Exception as e:
        st.error(f"❌ Error loading metadata: {str(e)}")
        return None

@st.cache_resource
def load_vit():
    """Load ViT model"""
    try:
        vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        vit.head = nn.Identity()
        vit.to(device)
        vit.eval()
        st.success("✅ ViT loaded")
        return vit
        
    except Exception as e:
        st.error(f"❌ Error loading ViT: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image to 224x224"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_resized = image.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)
    
    return img_tensor.to(device), img_resized

def predict(image, model, vit, X, edge_index, metadata):
    """Predict cancer type"""
    
    # Preprocess
    img_tensor, img_resized = preprocess_image(image)
    
    # Extract features
    with torch.no_grad():
        features = vit(img_tensor).cpu().numpy()
    
    # k-NN
    KNN_K = metadata['knn_k']
    sims = cosine_similarity(features, X)[0]
    neighbors = sims.argsort()[-KNN_K:]
    
    # Build graph
    new_node = len(X)
    old_edges = edge_index.cpu().numpy()
    new_edges = np.vstack([np.repeat(new_node, KNN_K), neighbors])
    all_edges = np.hstack([old_edges, new_edges, new_edges[::-1]])
    graph_edges = torch.LongTensor(all_edges).to(device)
    
    # Add features
    all_features = torch.FloatTensor(np.vstack([X, features])).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(all_features, graph_edges)
        probs = torch.softmax(output[new_node], dim=0).cpu().numpy()
    
    pred_idx = probs.argmax()
    pred_class = metadata['class_names'][pred_idx]
    confidence = probs[pred_idx]
    
    return pred_class, confidence, probs, img_resized

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Cancer Detection", page_icon="🔬", layout="wide")

st.title("🔬 Cancer Detection System")
st.markdown("### AI-Powered Histopathology Image Classification")
st.markdown("---")

# Load all components
with st.spinner("Loading model components..."):
    model, checkpoint = load_model()
    X = load_features()
    edge_index = load_graph()
    metadata = load_metadata()
    vit = load_vit()

# Check if everything loaded
if None in [model, X, edge_index, metadata, vit]:
    st.error("❌ Failed to load model components. Check files and restart.")
    st.stop()

# Sidebar info
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("Test Accuracy", f"{checkpoint['best_val_acc']:.2%}")
    st.markdown("**Cancer Types:**")
    for cls in metadata['class_names']:
        st.markdown(f"- {cls}")

# Main UI
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    uploaded = st.file_uploader("Choose image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Original", use_container_width=True)

with col2:
    st.header("📊 Results")
    
    if uploaded:
        with st.spinner("Analyzing..."):
            try:
                pred_class, confidence, probs, img_resized = predict(
                    image, model, vit, X, edge_index, metadata
                )
                
                st.image(img_resized, caption="Preprocessed (224×224)", width=224)
                
                st.markdown("### 🎯 Prediction")
                if confidence > 0.90:
                    st.success(f"**{pred_class}**")
                elif confidence > 0.70:
                    st.warning(f"**{pred_class}**")
                else:
                    st.error(f"**{pred_class}**")
                
                st.metric("Confidence", f"{confidence:.2%}")
                
                st.markdown("### 📈 Probabilities")
                for i, cls in enumerate(metadata['class_names']):
                    st.progress(float(probs[i]), text=f"{cls}: {probs[i]:.2%}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")
st.warning("⚠️ For research purposes only. Consult medical professionals for diagnosis.")
