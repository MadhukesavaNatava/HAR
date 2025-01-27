import streamlit as st
import cv2
import os
import tempfile
from PIL import Image
import numpy as np

# Create a folder for selected frames if it doesn't exist
os.makedirs("selected_frames", exist_ok=True)


# Helper function: Save uploaded file to a temporary directory
def save_uploaded_file(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    return temp_file.name


# Helper function: Extract video metadata
def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"total_frames": total_frames, "fps": fps}


# Save the selected frame to a folder
def save_selected_frame(frame, frame_number):
    frame_path = f"selected_frames/frame_{frame_number}.jpg"
    cv2.imwrite(frame_path, frame)
    return frame_path


# Process the saved frame or image with a dummy model (replace with actual model logic)
def process_frame_with_model(file_path):
    # Load the image (frame or uploaded image)
    frame = cv2.imread(file_path)
    # Preprocessing (e.g., resizing or normalization)
    processed_frame = cv2.resize(frame, (224, 224))  # Example size for models like MobileNet or ResNet
    # Dummy result (replace this with actual model inference)
    result = {"prediction": "Dancing", "confidence": 0.95}
    return result


# Upload Page
def upload_page():
    st.title("Upload Video or Image")
    option = st.radio("Choose the type of file to upload:", ("Video", "Image"))

    if option == "Video":
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if video_file:
            video_path = save_uploaded_file(video_file)
            st.video(video_path)

            metadata = get_video_metadata(video_path)
            st.write(f"Total Frames: {metadata['total_frames']}, FPS: {metadata['fps']}")

            st.session_state["video_path"] = video_path
            st.session_state["file_type"] = "video"

            if st.button("Next"):
                st.session_state["page"] = "playback_controls"

    elif option == "Image":
        image_file = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])
        if image_file:
            # Save and display the image
            image_path = save_uploaded_file(image_file)
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.session_state["image_path"] = image_path
            st.session_state["file_type"] = "image"

            if st.button("Next"):
                st.session_state["page"] = "image_prediction"


# Playback Controls Page
def playback_controls_page():
    st.title("Video Playback Controls")
    video_path = st.session_state.get("video_path")

    if video_path:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if "current_frame" not in st.session_state:
            st.session_state["current_frame"] = 0

        # Slider for selecting frames
        st.session_state["current_frame"] = st.slider(
            "Select Frame", 0, total_frames - 1, st.session_state["current_frame"]
        )

        # Display the selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["current_frame"])
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Frame {st.session_state['current_frame']}")

            # Save and process the selected frame
            if st.button("Select This Frame"):
                frame_path = save_selected_frame(frame, st.session_state["current_frame"])
                st.success(f"Frame saved to {frame_path}")

                # Process the frame with the model
                result = process_frame_with_model(frame_path)
                st.write("Model Prediction:", result)
        else:
            st.error("Error reading the frame.")

        # Back button
        if st.button("Back"):
            st.session_state["page"] = "upload"

        cap.release()


# Image Prediction Page
def image_prediction_page():
    st.title("Image Prediction")
    image_path = st.session_state.get("image_path")

    if image_path:
        # Process the uploaded image with the model
        result = process_frame_with_model(image_path)
        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        st.write("Model Prediction:", result)

        # Back button
        if st.button("Back"):
            st.session_state["page"] = "upload"


# Main App Logic
if "page" not in st.session_state:
    st.session_state["page"] = "upload"

if st.session_state["page"] == "upload":
    upload_page()
elif st.session_state["page"] == "playback_controls":
    playback_controls_page()
elif st.session_state["page"] == "image_prediction":
    image_prediction_page()