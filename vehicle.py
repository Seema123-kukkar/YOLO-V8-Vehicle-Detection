import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🚗 Vehicle Detection App")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    results = model(image)
    
    plotted_img = results[0].plot()
    st.image(plotted_img, caption="Detection Result")
