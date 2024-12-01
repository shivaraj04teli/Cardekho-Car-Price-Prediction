import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
try:
    model = joblib.load("random_forest_model.pkl")
except:
    st.error("Please ensure the model file 'random_forest_model.pkl' is in the same directory as this script")
    st.stop()

# Set page config
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Title and description
st.title("🚗 Car Price Prediction App")
st.write("""
This application predicts car prices based on various features. Please input the car details in the sidebar.
""")

def user_input_features():
    st.sidebar.header("Car Details")
    
    # Year
    year = st.sidebar.slider("Year", 1990, 2024, 2020)
    
    # Mileage
    mileage = st.sidebar.number_input("Mileage (miles)", 
                                     min_value=0, 
                                     max_value=500000, 
                                     value=50000)
    
    # Make
    make = st.sidebar.selectbox("Make", 
                               ["Toyota", "Honda", "Ford", "BMW", "Mercedes", 
                                "Audi", "Chevrolet", "Nissan", "Hyundai", "Kia"])
    
    # Fuel Type
    fuel_type = st.sidebar.selectbox("Fuel Type", 
                                    ["Gasoline", "Diesel", "Electric", "Hybrid"])
    
    # Transmission
    transmission = st.sidebar.selectbox("Transmission", 
                                      ["Automatic", "Manual"])
    
    # Engine Size
    engine_size = st.sidebar.slider("Engine Size (L)", 
                                   min_value=1.0, 
                                   max_value=6.0, 
                                   value=2.0, 
                                   step=0.1)
    
    # Body Style
    body_style = st.sidebar.selectbox("Body Style", 
                                     ["Sedan", "SUV", "Coupe", "Truck", "Van"])
    
    # Number of Doors
    doors = st.sidebar.selectbox("Number of Doors", [2, 4, 5])
    
    data = {
        'year': year,
        'mileage': mileage,
        'make': make,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'engine_size': engine_size,
        'body_style': body_style,
        'doors': doors
    }
    return pd.DataFrame([data])

# Get user inputs
input_df = user_input_features()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.write("### Input Features")
    st.write(input_df)

# Add some basic input validation
if input_df['mileage'].iloc[0] > 0 and input_df['year'].iloc[0] >= 1990:
    # Make prediction
    if st.button("Predict Price", type="primary"):
        try:
            # Add a loading spinner
            with st.spinner('Calculating prediction...'):
                prediction = model.predict(input_df)
                
            # Display prediction
            with col2:
                st.write("### Prediction Results")
                st.success(f"Predicted Car Price: ${prediction[0]:,.2f}")
                
                # Add confidence note
                st.info("""
                Note: This prediction is based on historical data and market trends. 
                Actual prices may vary based on additional factors like:
                - Local market conditions
                - Car condition
                - Additional features
                - Color and interior options
                """)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please ensure all input features are in the correct format and try again.")
else:
    st.warning("Please ensure mileage is greater than 0 and year is 1990 or later.")

# Add footer with additional information
st.markdown("""---""")
st.markdown("""
<div style='text-align: center'>
    <small>
        This model is trained on historical car price data. 
        Results should be used as estimates only.
    </small>
</div>
""", unsafe_allow_html=True) 