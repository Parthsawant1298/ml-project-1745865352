import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the model from file
def load_model():
  try:
      with open('best_model.pkl', 'rb') as f:
          model = pickle.load(f)
      return model
  except Exception as e:
      st.error(f"Error loading model: {e}")
      return None

# Streamlit UI for predictions
def main():
  st.title("Model Prediction App")
  
  model = load_model()
  if not model:
      st.stop()
  
  # Display information about the model
  st.write("## Model Information")
  model_type = type(model).__name__
  st.write(f"Model type: {model_type}")
  
  # Create input fields for each feature
  st.write("## Enter Feature Values")
  
  # Get user inputs
  user_inputs = {
    'Age': st.number_input('Enter Age', value=0.0),
    'Sex': st.number_input('Enter Sex', value=0.0),
    'Chest pain type': st.number_input('Enter Chest pain type', value=0.0),
    'BP': st.number_input('Enter BP', value=0.0),
    'Cholesterol': st.number_input('Enter Cholesterol', value=0.0),
    'FBS over 120': st.number_input('Enter FBS over 120', value=0.0),
    'EKG results': st.number_input('Enter EKG results', value=0.0),
    'Max HR': st.number_input('Enter Max HR', value=0.0),
    'Exercise angina': st.number_input('Enter Exercise angina', value=0.0),
    'ST depression': st.number_input('Enter ST depression', value=0.0),
    'Slope of ST': st.number_input('Enter Slope of ST', value=0.0),
    'Number of vessels fluro': st.number_input('Enter Number of vessels fluro', value=0.0),
    'Thallium': st.number_input('Enter Thallium', value=0.0),

  }
  
  # Predict the output
  if st.button("Predict"):
      try:
          # Create a DataFrame with the input values
          input_df = pd.DataFrame([user_inputs])
          
          # Make prediction
          prediction = model.predict(input_df)
          
          # Display the prediction
          st.write("## Prediction Result")
          
          # Check if it's a classification or regression model
          if hasattr(model, 'classes_'):
              # Classification model
              st.write(f"Predicted class: {prediction[0]}")
              
              # If model has predict_proba method, show probabilities
              if hasattr(model, 'predict_proba'):
                  try:
                      proba = model.predict_proba(input_df)
                      st.write("### Class Probabilities")
                      for i, class_name in enumerate(model.classes_):
                          st.write(f"{class_name}: {proba[0][i]:.4f}")
                  except:
                      pass
          else:
              # Regression model
              st.write(f"Predicted value: {prediction[0]:.4f}")
              
      except Exception as e:
          st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
  main()