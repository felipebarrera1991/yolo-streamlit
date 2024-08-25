import os
import torch
import streamlit as st
from model import load_model
from input import image_input, video_input

# Set Streamlit page layout to wide
st.set_page_config(layout="wide")

def main():
    """
    Main function for the Object Recognition Dashboard.
    
    Allows users to upload YOLOv8 or YOLOv10 models, configure the device (CPU or CUDA),
    and process input from images or videos for object detection.
    """
    st.title("Object Recognition Dashboard")
    st.sidebar.title("Settings")

    # Select YOLO model version
    model_src = st.sidebar.radio("Select YOLOv8 weight file", ["YOLOv8", "YOLOv10"])

    # Set model path based on user selection
    if model_src == "YOLOv8":
        cfg_model_path = '../models/yolov8n.pt'
    else:
        cfg_model_path = '../models/yolov10n.pt'

    st.sidebar.text(cfg_model_path.split("/")[-1])
    st.sidebar.markdown("---")

    # Check if the model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available! Please add it to the models folder.", icon="⚠️")
    else:
        # Check if CUDA is available and allow the user to select the device
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # Load the YOLO model
        model = load_model(cfg_model_path, device_option)

        # Confidence threshold slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)

        st.sidebar.markdown("---")

        # Input type selection: image or video
        input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

        # Process input based on the selected type
        if input_option == 'image':
            image_input(model, confidence)
        else:
            video_input(model, confidence)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass