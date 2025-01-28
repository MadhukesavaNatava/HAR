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
    duration = total_frames / fps  # Calculate video duration in seconds
    cap.release()
    return {"total_frames": total_frames, "fps": fps, "duration": duration}


# Extract frames every 6 seconds
def extract_frames(video_path, interval=6):
    """Extract frames from the video every 'interval' seconds."""
    metadata = get_video_metadata(video_path)
    fps = metadata["fps"]
    total_frames = metadata["total_frames"]
    duration = metadata["duration"]

    # Calculate frame numbers to extract
    frame_numbers = [int(fps * i) for i in range(0, int(duration), interval)]

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    extracted_frames = []

    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frame_path = f"selected_frames/frame_{frame_number}.jpg"
            cv2.imwrite(frame_path, frame)  # Save the frame
            extracted_frames.append(frame_path)

    cap.release()
    return extracted_frames  # Return paths of extracted frames


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
            st.write(
                f"Total Frames: {metadata['total_frames']}, FPS: {metadata['fps']}, Duration: {metadata['duration']} seconds")

            st.session_state["video_path"] = video_path
            st.session_state["file_type"] = "video"

            if st.button("Extract Frames"):
                extracted_frames = extract_frames(video_path, interval=6)
                st.session_state["extracted_frames"] = extracted_frames
                st.success(f"Extracted {len(extracted_frames)} frames every 6 seconds.")
                st.session_state["page"] = "frame_review"

    elif option == "Image":
        image_file = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])
        if image_file:
            # Save and display the image
            image_path = save_uploaded_file(image_file)
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Store image path in session state for further processing
            st.session_state["image_path"] = image_path
            st.session_state["file_type"] = "image"

            if st.button("Next"):
                st.session_state["page"] = "image_prediction"


# Frame Review Page
def frame_review_page():
    st.title("Review Extracted Frames")

    if "extracted_frames" in st.session_state:
        frames = st.session_state["extracted_frames"]

        # Display frames and allow the user to select one for prediction
        selected_frame = st.selectbox("Select a frame for prediction:", frames)
        if selected_frame:
            st.image(selected_frame, caption=f"Selected Frame: {os.path.basename(selected_frame)}")

            if st.button("Predict"):
                # Use the model to predict on the selected frame
                result = process_frame_with_model(selected_frame)
                st.write("Model Prediction:", result)

        # Add a Back button to return to the main upload page
        if st.button("Back to Main"):
            st.session_state["page"] = "upload"
    else:
        st.error("No frames found. Please upload a video and extract frames first.")
        if st.button("Back"):
            st.session_state["page"] = "upload"


# Image Prediction Page
def image_prediction_page():
    st.title("Image Prediction")
    image_path = st.session_state.get("image_path")

    if image_path:
        # Process the uploaded image with the model
        result = process_frame_with_model(image_path)
        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        st.write("Model Prediction:", result)

        if st.button("Back"):
            st.session_state["page"] = "upload"


# Main App Logic
if "page" not in st.session_state:
    st.session_state["page"] = "upload"

if st.session_state["page"] == "upload":
    upload_page()
elif st.session_state["page"] == "frame_review":
    frame_review_page()
elif st.session_state["page"] == "image_prediction":
    image_prediction_page()

