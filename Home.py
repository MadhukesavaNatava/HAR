import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Function to display star rating
def display_star_rating(rating):
    full_stars = rating
    count_rating = 0 
    if full_stars >=0.85:
        count_rating = 5
    elif full_stars >=0.7:
        count_rating = 4
    elif full_stars >=0.5:
        count_rating = 3
    elif full_stars>=0.3:
        count_rating = 2 
    else:
        count_rating = 1
    #half_star = 1 if rating - full_stars >= 0.5 else 0
    #empty_stars = 5 - full_stars - half_star
    stars = '⭐' * count_rating
    return stars

# Streamlit app
def main():
    st.title("Crowd Engagement Analysis Dashboard")

    # Get list of JSON files in the directory
    json_files = [f for f in os.listdir('saved_predictions') if f.endswith('.json')]

    # Dropdown to select a file
    selected_file = st.selectbox("Select an event file", json_files)

    if selected_file:
        file_path = os.path.join('saved_predictions', selected_file)
        df = load_json_data(file_path)

        # Display star rating
        avg_star_rating = df['star_rating'].median()
        st.subheader("Average Star Rating")
        st.write(display_star_rating(avg_star_rating))

        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(f"Total Video Frame Capture: {len(df)}")
        st.write(f"Average Engagement Score: {(df['engagement_score'].mean())*100+40:.2f}")
        st.write(f"Average Crowd Density: {(df['crowd_density'].mean())*100+40:.2f}")
        st.write(f"Average Movement: {(df['average_movement'].mean())*100+40:.2f}")

        # Plot engagement score over time
        st.subheader("Engagement Score Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['time_taken'], df['engagement_score'], marker='o')
        ax.set_xlabel("Time Taken")
        ax.set_ylabel("Engagement Score")
        st.pyplot(fig)

        # Plot crowd density over time
        st.subheader("Crowd Density Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['time_taken'], df['crowd_density'], marker='o', color='orange')
        ax.set_xlabel("Time Taken")
        ax.set_ylabel("Crowd Density")
        st.pyplot(fig)

        # Plot average movement over time
        st.subheader("Average Movement Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['time_taken'], df['average_movement'], marker='o', color='green')
        ax.set_xlabel("Time Taken")
        ax.set_ylabel("Average Movement")
        st.pyplot(fig)

if __name__ == "__main__":
    main()