import streamlit as st
import pandas as pd
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

# Create a Streamlit web app
st.title(':red[Flight Fare Prediction]')

# Define a function for the prediction form
def prediction_form():
    st.write('Fill in the details to predict flight fare:')
    # Create input widgets for user input
    airline = st.selectbox('Airline', ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia','Vistara Premium economy', 'Jet Airways Business','Multiple carriers Premium economy', 'Trujet'])  
    source = st.selectbox('Source', ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']) 
    destination = st.selectbox('Destination', ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])  
    total_stops = st.number_input('Total Stops', min_value=0, max_value=5, value=1)
    journey_day = st.number_input('Journey Day', min_value=1, max_value=31, value=1)
    journey_month = st.number_input('Journey Month', min_value=1, max_value=12, value=1)
    journey_year = st.number_input('Journey Year', min_value=2000, max_value=2030, value=2022)
    hours = st.number_input('Departure Hours', min_value=0, max_value=23, value=0)
    minutes = st.number_input('Departure Minutes', min_value=0, max_value=59, value=0)
    arrival_hour = st.number_input('Arrival Hour', min_value=0, max_value=23, value=0)
    arrival_min = st.number_input('Arrival Minutes', min_value=0, max_value=59, value=0)
    duration_hours = st.number_input('Duration Hours', min_value=0, max_value=23, value=0)
    duration_mins = st.number_input('Duration Minutes', min_value=0, max_value=59, value=0)

    # Create a "Predict" button
    if st.button('Predict'):
        # Create a CustomData object from user input
        data = CustomData(
            Airline=airline,
            Source=source,
            Destination=destination,
            Total_Stops=total_stops,
            Journey_day=journey_day,
            Journey_month=journey_month,
            Journey_year=journey_year,
            hours=hours,
            minutes=minutes,
            Arrival_hour=arrival_hour,
            Arrival_min=arrival_min,
            duration_hours=duration_hours,
            duration_mins=duration_mins
        )
        pred_df = data.get_data_as_data_frame()

        # Make a prediction using the PredictPipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Display the prediction result
        st.subheader('Prediction Result:')
        st.write(f'Predicted Fare: Rs. {results[0]}')

# Display the prediction form
prediction_form()


