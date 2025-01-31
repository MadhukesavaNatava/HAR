import streamlit as st
import cv2
import os
import tempfile
from PIL import Image
import numpy as np
import validators
import yt_dlp

# Create a folder for selected frames if it doesn't exist
os.makedirs("selected_frames", exist_ok=True)

# User credentials (Modify here for authentication)
USER_CREDENTIALS = {"admin": "123"}  # 'admin' is the username, '123' is the password

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
    metadata = get_video_metadata(video_path)
    fps = metadata["fps"]
    duration = metadata["duration"]
    frame_numbers = [int(fps * i) for i in range(0, int(duration), interval)]

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
    return extracted_frames

# Process the saved frame with a dummy model (replace with actual model logic)
def process_frame_with_model(file_path):
    frame = cv2.imread(file_path)
    processed_frame = cv2.resize(frame, (224, 224))  # Example size for models like MobileNet or ResNet
    result = {"prediction": "Dancing", "confidence": 0.95}
    return result

# Helper function: Validate and download YouTube links using yt-dlp
def handle_video_link(video_link):
    if not validators.url(video_link):
        raise ValueError("Invalid URL provided.")

    if "youtube.com" in video_link or "youtu.be" in video_link:
        ydl_opts = {"format": "best"}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_link, download=False)
            return info["url"]
    else:
        return video_link

# Login Page
def login_page():
    st.title("User Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["page"] = "upload"  # Redirect to upload page after login
            st.success("Login successful! Redirecting to the app...")
        else:
            st.error("Invalid username or password")

# Upload Page
def upload_page():
    st.title("Upload Video or Provide a Video Link")

    option = st.radio("Choose how you want to upload the video:", ("Upload a file", "Provide a video link"))

    if option == "Upload a file":
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        if video_file:
            video_path = save_uploaded_file(video_file)
            st.video(video_path)
            metadata = get_video_metadata(video_path)
            st.write(f"Total Frames: {metadata['total_frames']}, FPS: {metadata['fps']}, Duration: {metadata['duration']} seconds")
            st.session_state["video_path"] = video_path
            st.session_state["file_type"] = "file"

            if st.button("Extract Frames"):
                extracted_frames = extract_frames(video_path, interval=6)
                st.session_state["extracted_frames"] = extracted_frames
                st.success(f"Extracted {len(extracted_frames)} frames every 6 seconds.")
                st.session_state["page"] = "frame_review"

    elif option == "Provide a video link":
        video_link = st.text_input("Paste the video link (YouTube or direct URL):")

        if st.button("Process Video Link"):
            try:
                video_stream_url = handle_video_link(video_link)
                st.session_state["video_path"] = video_stream_url
                st.session_state["file_type"] = "link"
                st.success("Video link processed successfully! You can now extract frames.")
                st.video(video_stream_url)
            except Exception as e:
                st.error(f"Error processing video link: {e}")

        if st.session_state.get("video_path"):
            if st.button("Extract Frames"):
                extracted_frames = extract_frames(st.session_state["video_path"], interval=6)
                st.session_state["extracted_frames"] = extracted_frames
                st.success(f"Extracted {len(extracted_frames)} frames every 6 seconds.")
                st.session_state["page"] = "frame_review"

# Frame Review Page
def frame_review_page():
    st.title("Review Extracted Frames")

    if "extracted_frames" in st.session_state:
        frames = st.session_state["extracted_frames"]
        selected_frame = st.selectbox("Select a frame for prediction:", frames)

        if selected_frame:
            st.image(selected_frame, caption=f"Selected Frame: {os.path.basename(selected_frame)}")

            if st.button("Predict"):
                result = process_frame_with_model(selected_frame)
                st.write("Model Prediction:", result)

    if st.button("Back to Main"):
        st.session_state["page"] = "upload"

# Logout Function
def logout():
    st.session_state["authenticated"] = False
    st.session_state["page"] = "login"

# Main App Logic
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# If the user is not authenticated, show the login page
if not st.session_state["authenticated"]:
    login_page()
else:
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        logout()

    # Page navigation
    if st.session_state["page"] == "upload":
        upload_page()
    elif st.session_state["page"] == "frame_review":
        frame_review_page()
