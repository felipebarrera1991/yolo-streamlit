import cv2
import streamlit as st
import time
from PIL import Image
from model import infer_image

def image_input(model, confidence):
    """
    Handles image input for the Streamlit dashboard.

    Parameters:
        model: The YOLO model used for inference.
        confidence: Confidence threshold for object detection.
    """
    img_file = None
    img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])

    if img_bytes:
        # Save the uploaded image to the specified directory
        img_file = "../data/image/upload." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(model, img_file, confidence)
            st.image(img, caption="Model Prediction")

def video_input(model, confidence):
    """
    Handles video input for the Streamlit dashboard.

    Parameters:
        model: The YOLO model used for inference.
        confidence: Confidence threshold for object detection.
    """
    vid_file = None

    # Select input source: sample data or uploaded video
    data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

    if data_src == 'Sample data':
        vid_file = "../data/video/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            # Save the uploaded video to the specified directory
            vid_file = "../data/video/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break

            # Resize and convert the frame for display
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(model, frame, confidence)
            output.image(output_img)

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()