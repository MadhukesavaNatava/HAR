# HAR


  # CROWD ENGAGEMENT ANALYSIS

## Overview
Live Video Interaction Detector is a **Streamlit-based web application** designed for real-time video analysis. Users can:
- Upload a video or provide a YouTube link for analysis.
- Analyze human interaction levels using an AI model.
- Visualize engagement levels through dynamic graphs.
- Process video frames at 1 frame per second for better accuracy.

## Features
- Upload local videos (`.mp4`, `.avi`, `.mov`)
- Analyze YouTube videos via URL
- Real-time frame processing with AI
- Dynamic graph visualizations using Plotly
- User login authentication system
- Customizable AI model integration

## Installation
   Install Required Dependencies
     
      pip install -r requirements.txt


## Run the Streamlit App

      streamlit run app.py


## Project Structure

      frontend-streamlit/
        │── app.py                    # Main Streamlit App
        │── requirements.txt          # Dependencies
        │── README.md                 # Project Documentation
        │── frontend/
        │   │── action_predictor.py   # AI Model Processing (if applicable)
        │   │── label_map.txt         # Model Labels
        │   └── models/
        │       └── saved_model.pb    # Trained AI Model




## Future Improvements

  Optimize performance using multithreading
  
  Integrate a real-time AI model for live event tracking



