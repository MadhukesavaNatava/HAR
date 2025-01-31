import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.express as px
from action_predictor import ActionPredictor
import json  # For saving predictions locally
from pathlib import Path  # For handling file paths
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Constants
I3D_FRAME_SIZE = (224, 224)
MAX_FRAMES = 32
PREDICTION_INTERVAL = 1.0  # Predict every 1 second
WARMUP_FRAMES = 10  # Warm-up frames to ensure system is ready

# Define high-engagement and movement-related actions
HIGH_ENGAGEMENT_ACTIONS = ["celebrating", "cheerleading", "dancing", "robot dancing", "high kick", "zumba", "singing", "jumping"]
MOVEMENT_ACTIONS = ["celebrating", "cheerleading", "dancing", "robot dancing", "high kick", "zumba", "marching", "jumping"]

# Set page config as the first command
st.set_page_config(page_title="Crowd Engagement Analysis", layout="wide")

# Load models
@st.cache_resource
def load_models():
    i3d_model_path = "../model/i3d/"
    label_map_path = "../model/i3d/label_map.txt"
    return ActionPredictor(i3d_model_path, label_map_path)

# Initialize the action predictor
action_predictor = load_models()

# Function to save predictions locally as JSON
def save_predictions_locally(predictions, project_title):
    """Save predictions to a local JSON file."""
    # Create a directory for saved predictions if it doesn't exist
    try:
        save_dir = Path("saved_predictions")
        save_dir.mkdir(exist_ok=True)
        # Define the file path
        file_path = save_dir / f"{project_title.replace(' ', '_').lower()}.json"
    
        # Save predictions as JSON
        with open(file_path, "w") as f:
            json.dump(predictions, f, indent=4)
    
        st.success(f"Predictions saved locally as: {file_path}")
    except Exception as e:
        print('error on save, ', e)

# Function to generate PDF
def generate_pdf(star_rating, overall_engagement, overall_crowd_density, overall_average_movement, df_detailed):
    """Generate a PDF with the analysis results."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add content to the PDF
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Crowd Engagement Analysis Report")
    c.setFont("Helvetica", 12)

    # Star Rating
    c.drawString(50, height - 80, f"Event Star Rating: {star_rating}")

    # Overall Scores
    c.drawString(50, height - 110, "Overall Scores:")
    c.drawString(70, height - 130, f"Engagement Score: {overall_engagement:.2f}")
    c.drawString(70, height - 150, f"Crowd Density: {overall_crowd_density:.2f}")
    c.drawString(70, height - 170, f"Average Movement: {overall_average_movement:.2f}")

    # Detailed Predictions Table
    c.drawString(50, height - 200, "Detailed Predictions:")
    table_data = [["Time (seconds)", "Top Action", "Confidence"]]
    for _, row in df_detailed.iterrows():
        table_data.append([row["Time (seconds)"], row["Top Action"], f"{row['Confidence']:.2f}"])

    # Draw the table
    x_start = 50
    y_start = height - 230
    row_height = 20
    col_widths = [100, 100, 100]

    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            c.drawString(x_start + sum(col_widths[:j]), y_start - i * row_height, str(cell))

    # Save the PDF
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

class ActionRecognitionApp:
    def __init__(self):
        self.init_session_states()

    def init_session_states(self):
        """Initialize session state variables."""
        session_vars = {
            'upload_video_source': None,
            'upload_timeline': [],
            'upload_video_duration': 0.0,
            'upload_predictions_ready': False,
            'url_video_source': None,
            'url_timeline': [],
            'url_video_duration': 0.0,
            'url_predictions_ready': False,
            'video_url': None,
            'current_tab': None  # Track the current tab for predictions
        }
        for key, value in session_vars.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def video_upload(self):
        """Handle video upload and display."""
        st.subheader("Upload Video")
        uploaded_file = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'], key="upload_video")
        
        if uploaded_file is not None:
            self.save_uploaded_video(uploaded_file)
            self.display_video("upload")
            
            if st.button("Predict Actions", key="predict_upload"):
                st.session_state.current_tab = "upload"
                self.predict_actions("upload")

    def load_video_from_url(self):
        """Load video from URL."""
        st.subheader("Load Video from URL")
        video_url = st.text_input("Enter video URL:", key="video_url")
        
        if video_url:
            st.session_state.url_video_source = video_url
            st.session_state.video_url = video_url
            self.display_video("url")
            
            if st.button("Predict Actions", key="predict_url"):
                st.session_state.current_tab = "url"
                self.predict_actions("url")

    def save_uploaded_video(self, uploaded_file):
        """Save the uploaded video to a temporary file."""
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.upload_video_source = "temp_video.mp4"

    def display_video(self, tab):
        """Display the uploaded video."""
        if tab == "upload" and st.session_state.upload_video_source:
            video_file = open("temp_video.mp4", "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)
        elif tab == "url" and st.session_state.url_video_source:
            st.video(st.session_state.url_video_source)

    def predict_actions(self, tab):
        """Predict actions from the uploaded video and display predictions dynamically."""
        if tab == "upload" and not st.session_state.upload_video_source:
            st.error("No video uploaded.")
            return
        elif tab == "url" and not st.session_state.url_video_source:
            st.error("No video URL provided.")
            return

        video_source = st.session_state.upload_video_source if tab == "upload" else st.session_state.url_video_source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Error: Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = float(total_frames / fps)

        # Warm-up phase
        for _ in range(WARMUP_FRAMES):
            ret, _ = cap.read()
            if not ret:
                st.error("Error: Video is too short for warm-up.")
                return

        frame_buffer = []
        timeline = []
        frame_count = WARMUP_FRAMES

        # Placeholder for displaying instant predictions
        prediction_placeholder = st.empty()

        with st.spinner("Predicting actions..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                elapsed_time = frame_count / fps
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > MAX_FRAMES:
                    frame_buffer.pop(0)

                if elapsed_time % PREDICTION_INTERVAL < (1 / fps):
                    i3d_actions = {}
                    if len(frame_buffer) >= MAX_FRAMES:
                        i3d_actions = action_predictor.predict_i3d(frame_buffer, I3D_FRAME_SIZE, MAX_FRAMES)

                    timeline.append({
                        'time_taken': elapsed_time,
                        'i3d_actions': i3d_actions
                    })

                    # Display the latest prediction dynamically
                    if i3d_actions:
                        total_labels = 10  # Total predicted labels
                        total_score = sum(i3d_actions.get(action, 0) for action in HIGH_ENGAGEMENT_ACTIONS)
                        engagement_score = (total_score / total_labels) * 100  # Normalize to 100%
                        engagement_color = "green" if engagement_score >= 3 else "orange" if engagement_score >= 1 else "grey"
                        engagement_val = "Great engagement level !!!" if engagement_score >= 3 else "Good, engagement level" if engagement_score >= 1 else "Engagement level is low !!"
                        prediction_placeholder.markdown(f"Prediction at {elapsed_time:.2f}s:<div style='color:{engagement_color}'> {engagement_val}</div>", unsafe_allow_html=True)
                frame_count += 1
        cap.release()

        if tab == "upload":
            st.session_state.upload_timeline = timeline
            st.session_state.upload_video_duration = video_duration
            st.session_state.upload_predictions_ready = True
        elif tab == "url":
            st.session_state.url_timeline = timeline
            st.session_state.url_video_duration = video_duration
            st.session_state.url_predictions_ready = True

        st.success("Predictions complete!")

    def display_analysis_results(self):
        """Display analysis results in the dedicated tab."""
        if st.session_state.current_tab == "upload" and not st.session_state.upload_predictions_ready:
            st.warning("No predictions available. Please upload a video and predict actions first.")
            return
        elif st.session_state.current_tab == "url" and not st.session_state.url_predictions_ready:
            st.warning("No predictions available. Please provide a video URL and predict actions first.")
            return

        timeline = (
            st.session_state.upload_timeline
            if st.session_state.current_tab == "upload"
            else st.session_state.url_timeline
        )

        # Define high-engagement and movement-related actions
        HIGH_ENGAGEMENT_ACTIONS = ["celebrating", "cheerleading", "dancing", "robot dancing", "high kick", "zumba", "singing","jumping"]
        MOVEMENT_ACTIONS = ["celebrating", "cheerleading", "dancing", "robot dancing", "high kick", "zumba", "marching", "jumping"]

        # Calculate overall metrics
        engagement_scores = []
        crowd_density_scores = []
        average_movement_scores = []

        for prediction in timeline:
            if prediction['i3d_actions']:
                actions = prediction['i3d_actions']

                # Engagement score
                engagement_score = sum(
                    confidence for action, confidence in actions.items()
                    if action in HIGH_ENGAGEMENT_ACTIONS
                )
                engagement_scores.append(engagement_score)

                # Crowd density
                crowd_density = sum(actions.values()) / len(actions)
                crowd_density_scores.append(crowd_density)

                # Average movement
                movement_confidences = [
                    confidence for action, confidence in actions.items()
                    if action in MOVEMENT_ACTIONS
                ]
                average_movement = sum(movement_confidences) / len(movement_confidences) if movement_confidences else 0
                average_movement_scores.append(average_movement)

        # Overall scores
        overall_engagement = np.mean(engagement_scores)
        overall_crowd_density = np.mean(crowd_density_scores)
        overall_average_movement = np.mean(average_movement_scores)

        overall_score = (
        0.5 * overall_engagement +  # Weight for engagement
        0.3 * overall_crowd_density +  # Weight for crowd density
        0.2 * overall_average_movement  # Weight for average movement
    )

        # Map overall score to star rating (1 to 5 stars)
        overall_score = overall_score * 4
        st.subheader("Event Engagement Rating")
        # Display star rating at the beginning
        if overall_score >= 0.9:
            st.markdown(" ⭐⭐⭐⭐⭐")
        elif overall_score >= 0.7:
            
            st.markdown("⭐⭐⭐⭐")  # 4 stars
        elif overall_score >= 0.5:
            st.markdown("⭐⭐⭐")  # 3 stars
        elif overall_score >= 0.3:
            st.markdown("⭐⭐")  # 2 stars
        else:
            st.markdown("⭐")  # 1 star

        # Display overall score in an interactive way
        st.subheader("Overall Video Score")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_engagement = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_engagement * 100 + 45,
                title={"text": "Engagement Score"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "green" if overall_engagement > 75 else "yellow" if overall_engagement > 45 else "red"},
                       "steps": [
                           {"range": [0, 40], "color": "red"},
                           {"range": [40, 70], "color": "yellow"},
                           {"range": [70, 100], "color": "green"}
                       ]}
            ))
            st.plotly_chart(fig_engagement)
    
        with col2:
            fig_density = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_crowd_density * 100 + 45,
                title={"text": "Crowd Density"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "green" if overall_crowd_density > 75 else "yellow" if overall_crowd_density > 45 else "red"},
                       "steps": [
                           {"range": [0, 40], "color": "red"},
                           {"range": [40, 70], "color": "yellow"},
                           {"range": [70, 100], "color": "green"}
                       ]}
            ))
            st.plotly_chart(fig_density)
    
        with col3:
            fig_movement = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_average_movement * 100 + 45,
                title={"text": "Average Movement"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "green" if overall_average_movement > 70 else "yellow" if overall_average_movement > 40 else "red"},
                       "steps": [
                           {"range": [0, 40], "color": "red"},
                           {"range": [40, 70], "color": "yellow"},
                           {"range": [70, 100], "color": "green"}
                       ]}
            ))
            st.plotly_chart(fig_movement)
        # Display overall score in an interactive way
        # Prepare data for table and graphs
        time_series_data = []
        for prediction in timeline:
            if prediction['i3d_actions']:
                actions = prediction['i3d_actions']
                time_taken = prediction['time_taken']

                # Engagement score
                engagement_score = sum(
                    confidence for action, confidence in actions.items()
                )

                # Crowd density
                crowd_density = sum(actions.values()) / len(actions)

                # Average movement
                movement_confidences = [
                    confidence for action, confidence in actions.items()
                ]
                average_movement = sum(movement_confidences) / len(movement_confidences) if movement_confidences else 0

                # Store data for table
                time_series_data.append({
                    'time_taken': time_taken,
                    'action': max(actions, key=actions.get),
                    'confidence': actions[max(actions, key=actions.get)],
                    'engagement_score': engagement_score,
                    'crowd_density': crowd_density,
                    'average_movement': average_movement,
                    'star_rating': overall_score
                })

        df = pd.DataFrame(time_series_data)

        # Display detailed predictions in a table
        #st.dataframe(timeline)
        # Prepare data for detailed predictions graph
        # Prepare data for detailed predictions graph
        detailed_data = []
        for prediction in timeline:
            if prediction['i3d_actions']:
                actions = prediction['i3d_actions']
                time_taken = prediction['time_taken']
        
                # Get the top predicted action and its confidence
                top_action = max(actions, key=actions.get)
                top_confidence = actions[top_action]
        
                detailed_data.append({
                    'Time (seconds)': time_taken,
                    'Top Action': top_action,
                    'Confidence': top_confidence * 100
                })
        
        df_detailed = pd.DataFrame(detailed_data)
        
        # Display detailed predictions as an interactive line chart
        st.subheader("Detailed Predictions Over Time")
        if not df_detailed.empty:
            # Add a color column based on the confidence threshold
            df_detailed['Color'] = df_detailed['Confidence'].apply(lambda x: 'red' if x < 0.1 else 'green')
        
            # Create the line chart
            fig = px.line(
                df_detailed,
                x='Time (seconds)',
                y='Confidence',
                color='Color',  # Use the color column for conditional coloring
                title="Top Predicted Actions Over Time",
                labels={'Confidence': 'Confidence Score', 'Time (seconds)': 'Time (seconds)'},
                hover_data=['Top Action', 'Confidence']
            )
        
            # Fill the area under the line with color
            fig.update_traces(
                fill='tozeroy',  # Fill the area under the line
                line=dict(width=2),  # Set line width
                mode='lines'  # Display only lines (no markers)
            )
        
            # Update layout for better visualization
            fig.update_layout(
                showlegend=False,  # Hide the legend (since colors are self-explanatory)
                xaxis_title="Time (seconds)",
                yaxis_title="Confidence Score",
                hovermode="x unified"  # Show hover data for all points at the same x-value
            )
        
            # Display the chart
            st.plotly_chart(fig, use_container_width=True, key="detailed_predictions")
        # Visualizations
        st.subheader("Engagement Analysis")

        # Line chart for engagement over time
        if not df.empty:
            st.write("### Engagement Over Time")
            fig = px.line(df, x='time_taken', y='engagement_score', title="Engagement Over Time",
                          labels={'engagement_score': 'Engagement Score', 'time_taken': 'Time (seconds)'})
            st.plotly_chart(fig, use_container_width=True, key="engagement_over_time")

        # Bar chart for crowd density over time
        if not df.empty:
            st.write("### Crowd Density Over Time")
            fig = px.line(df, x='time_taken', y='crowd_density', title="Crowd Density Over Time",
                          labels={'crowd_density': 'Crowd Density', 'time_taken': 'Time (seconds)'})
            st.plotly_chart(fig, use_container_width=True, key="crowd_density_over_time")

        # Bar chart for average movement over time
        if not df.empty:
            st.write("### Average Movement Over Time")
            fig = px.line(df, x='time_taken', y='average_movement', title="Average Movement Over Time",
                          labels={'average_movement': 'Average Movement', 'time_taken': 'Time (seconds)'})
            st.plotly_chart(fig, use_container_width=True, key="average_movement_over_time")

        # Save predictions locally as JSON
        project_title = st.text_input("Enter Project Title for Saving Predictions", key="project_title")
        if st.button("Save Predictions Locally", key="save_predictions"):
            if project_title:
                save_predictions_locally(time_series_data, project_title)
            else:
                st.warning("Please enter a project title before saving.")

        # Export to PDF
        if st.button("📄 Export to PDF", key="export_pdf"):
            pdf_buffer = generate_pdf(
                star_rating="⭐⭐⭐⭐⭐" if overall_score >= 0.9 else "⭐⭐⭐⭐" if overall_score >= 0.7 else "⭐⭐⭐" if overall_score >= 0.5 else "⭐⭐" if overall_score >= 0.3 else "⭐",
                overall_engagement=overall_engagement*100,
                overall_crowd_density=overall_crowd_density*100,
                overall_average_movement=overall_average_movement*100,
                df_detailed=df_detailed
            )
            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name="crowd_engagement_analysis.pdf",
                mime="application/pdf",
                key="download_pdf"
            )

    def main(self):
        """Main application logic."""
        st.title("Crowd Engagement Analysis")
        
        # Create tabs for Upload Video, Load Video from URL, and Prediction Stats
        tab1, tab2, tab3 = st.tabs(["Upload Video", "Load Video from URL", "Prediction Stats"])
        
        with tab1:
            self.video_upload()
        
        with tab2:
            self.load_video_from_url()
        
        with tab3:
            self.display_analysis_results()

# Main function to run the app
def main():
    app = ActionRecognitionApp()
    app.main()

if __name__ == "__main__":
    main()