import os
import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Indoor Plant Disease Detector", page_icon="🌿")
st.title("🌿 High-Accuracy Indoor Plant Disease Detection")

MODEL_PATH = r"D:\AIML ptoject\saved_models\plant_disease_svm_pipeline.pkl"

@st.cache_resource
def load_pipeline(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

pipeline = load_pipeline(MODEL_PATH)
if pipeline is None:
    st.error(f"❌ Model file not found at `{MODEL_PATH}`. Please run your training script first.")
    st.stop()

# Extract pipeline components
svm_model = pipeline['model']
scaler = pipeline['scaler']
pca = pipeline['pca']
label_encoder = pipeline['label_encoder']
target_size = pipeline['target_size']

# ------------------------------------------------------------------------------
# 2. Robust HOG & Color Feature Extraction
# ------------------------------------------------------------------------------
def compute_hog_features(gray_img, win_size=(128, 128)):
    """Computes HOG features safely across any OpenCV version."""
    # Attempt native OpenCV HOGDescriptor
    if hasattr(cv2, 'HOGDescriptor'):
        hog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
        return hog.compute(gray_img).flatten()
    
    # Fallback: Manual HOG calculation using Sobel gradients
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Quantize angles into 9 bins (0-180 degrees)
    angle = np.mod(angle, 180)
    bin_width = 20
    bins = (angle // bin_width).astype(int) % 9
    
    hist_list = []
    cell_size = 8
    h, w = gray_img.shape
    for row in range(0, h, cell_size):
        for col in range(0, w, cell_size):
            mag_cell = magnitude[row:row+cell_size, col:col+cell_size]
            bin_cell = bins[row:row+cell_size, col:col+cell_size]
            cell_hist = np.zeros(9, dtype=np.float32)
            for b in range(9):
                cell_hist[b] = np.sum(mag_cell[bin_cell == b])
            hist_list.append(cell_hist)
            
    hog_feat = np.concatenate(hist_list)
    cv2.normalize(hog_feat, hog_feat)
    return hog_feat

def predict_plant_disease(pil_image):
    # Convert PIL Image to OpenCV BGR format
    img_rgb = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Resize image to model input dimensions (128x128)
    img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    
    # 2. Extract HOG texture features
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feat = compute_hog_features(gray_img, win_size=target_size)
    
    # 3. Extract HSV Color Histograms
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])
    
    color_feat = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    cv2.normalize(color_feat, color_feat)
    
    # 4. Concatenate HOG + Color Features
    combined_feat = np.hstack([hog_feat, color_feat]).reshape(1, -1)
    
    # 5. Transform & Predict
    scaled_feat = scaler.transform(combined_feat)
    pca_feat = pca.transform(scaled_feat)
    
    pred_idx = svm_model.predict(pca_feat)[0]
    return label_encoder.inverse_transform([pred_idx])[0]

# ------------------------------------------------------------------------------
# 3. Streamlit Interface
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Leaf Photo", type=["png", "jpg", "jpeg", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing plant features..."):
        prediction = predict_plant_disease(image)
        
    st.success("Diagnosis Complete!")
    st.metric("Predicted Disease", prediction)