# !pip install gradio

import gradio as gr
import pickle
import pandas as pd

# Load your model (replace with your actual model path)
pipe = pickle.load(open('/content/drive/MyDrive/Project/T20_run_prediction/pipe.pkl', 'rb'))

teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Johannesburg', 'Mirpur', 'Auckland', 'Cape Town', 'London',
    'Barbados', 'St Lucia', 'Wellington', 'Durban', 'Lauderhill',
    'Hamilton', 'Centurion', 'Abu Dhabi', 'Manchester', 'Mumbai',
    'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata',
    'Sydney', 'Delhi', 'Nagpur', 'Cardiff', 'Chandigarh', 'Lahore',
    'Bangalore', 'St Kitts', 'Christchurch', 'Trinidad'
]

def predict_score(batting_team, bowling_team, city,current_score, cuurent_runrate, overs, wickets, last_five):
    try:
        # Calculate additional features
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        current_runrate = current_score / overs 

        # Prepare input data
        input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'current_runrate': [current_runrate],
            'last_five': [last_five]
        })

        # Predict using the trained pipeline
        result = pipe.predict(input_data)
        return f"Predicted Score: {int(result[0])}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_score,
    inputs=[
        gr.Dropdown(choices=teams, label="Select Batting Team"),
        gr.Dropdown(choices=teams, label="Select Bowling Team"),
        gr.Dropdown(choices=cities, label="Select City"),
        gr.Number(label="Current Score"),
        gr.Number(label="Overs Done (works for over > 5)"),
        gr.Number(label="Wickets Out"),
        gr.Number(label="Runs Scored in Last 5 Overs")
    ],
    outputs="text",
    title="Cricket Score Predictor",
    description="Predict the final score of a T20 cricket match."
)

# Launch the Gradio interface
iface.launch(share=True)
