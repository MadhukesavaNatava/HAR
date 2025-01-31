import streamlit as st
import validators
import yt_dlp
import tempfile
import pandas as pd
import numpy as np
import time
import plotly.express as px

# User credentials
USER_CREDENTIALS = {"admin": "123"}

# Custom CSS for UI Enhancements
st.markdown("""
    <style>
        /* Background Styling */
        .main {
            background: linear-gradient(to right, #141E30, #243B55);
            color: white;
        }

        /* Centered login container */
        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
        }

        /* Login Box */
        .login-box {
            width: 400px;
            padding: 30px;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.8);
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.2);
            text-align: center;
        }

        /* Button Styling */
        .stButton>button {
            width: 100%;
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
        }

        /* Graph Container */
        .graph-container {
            display: flex;
            justify-content: center;
            gap: 40px;
            width: 100%;
            margin-top: 20px;
        }

        /* Graph Size */
        .stPlotlyChart div {
            width: 100% !important;
            max-width: 800px;
        }
    </style>
""", unsafe_allow_html=True)

# Dummy model function (Replace with actual AI model)
def process_frame_with_model():
    return np.random.uniform(0, 1)  # Simulated confidence score



# Helper function: Save uploaded file to a temporary directory
def save_uploaded_file(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    return temp_file.name

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

# Function to process and display graphs for both YouTube and uploaded videos
def process_video(video_url, video_type):
    st.subheader("Video Player")

    with st.spinner("Processing video... Please wait."):
        time.sleep(2)

    if video_type == "youtube":
        video_html = f"""
        <iframe width="100%" height="450" 
                src="{video_url}" 
                frameborder="0" allowfullscreen>
        </iframe>
        """
        st.components.v1.html(video_html, height=500)
    else:
        st.video(video_url)

    if st.button("Go to Main"):
        st.session_state["page"] = "main"
        st.rerun()

    duration = 20
    timestamps = []
    predictions = []

    st.markdown('<div class="graph-container">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Interaction Over Time")
        graph_placeholder1 = st.empty()

    with col2:
        st.subheader("Frame-wise Intensity")
        graph_placeholder2 = st.empty()

    st.markdown('</div>', unsafe_allow_html=True)

    start_time = time.time()
    for second in range(1, duration + 1):
        elapsed_time = time.time() - start_time

        interaction_score = process_frame_with_model()
        timestamps.append(elapsed_time)
        predictions.append(interaction_score)

        df = pd.DataFrame({"Time (s)": timestamps, "Score": predictions})

        fig1 = px.line(df, x="Time (s)", y="Score", markers=True, template="plotly_dark")
        fig1.update_layout(height=500, width=800)
        graph_placeholder1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(df, x="Time (s)", y="Score", template="plotly_dark",
                      color="Score", color_continuous_scale="viridis")
        fig2.update_layout(height=500, width=800)
        graph_placeholder2.plotly_chart(fig2, use_container_width=True)

        time.sleep(1)

# Login Page
def login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    st.title("User Login")

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["page"] = "main"
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main Page
def main_page():
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout", key="logout"):
        st.session_state["authenticated"] = False
        st.session_state["page"] = "login"
        st.rerun()

    st.sidebar.header("Options")
    option = st.sidebar.radio("Choose an action:", ["Upload Video", "Use YouTube Link"])

    if option == "Upload Video":
        st.subheader("Upload a Video File")
        video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

        if video_file:
            video_path = save_uploaded_file(video_file)
            if st.button("Start Analysis", key="start_analysis"):
                st.session_state["page"] = "analysis"
                st.session_state["video_path"] = video_path
                st.session_state["video_type"] = "upload"
                st.rerun()

    elif option == "Use YouTube Link":
        st.subheader("Provide a YouTube Video Link")
        video_link = st.text_input("Paste the YouTube link here:")

        if st.button("Process Video Link"):
            try:
                video_stream_url = handle_video_link(video_link)
                st.session_state["video_path"] = video_stream_url
                st.session_state["page"] = "analysis"
                st.session_state["video_type"] = "youtube"
                st.rerun()

            except Exception as e:
                st.error(f"Error processing video link: {e}")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login_page()
elif st.session_state["page"] == "main":
    main_page()
elif st.session_state["page"] == "analysis":
    process_video(st.session_state["video_path"], st.session_state["video_type"])
import tensorflow as tf
import streamlit as st

# Use st.cache to load the model only once
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.saved_model.load('C:/Users/moi/Downloads/saved_model.pb')
    return model

model = load_model()

# Assuming the model has a default serving signature
def predict(input_data):
    infer = model.signatures['serving_default']
    output = infer(tf.constant(input_data))['output_key']  # adjust 'output_key' based on your model's output
    return output.numpy()

# Add a button in Streamlit to make predictions
input_data = st.text_input("Enter input data here")
if st.button("Predict"):
    result = predict([input_data])  # Adjust input format based on your model needs
    st.write("Prediction:", result)
