import streamlit as st
import requests

# Streamlit app title
st.title("NLP Model Tester")

# Text input box
input_text = st.text_area("Enter some text:", "")

# Add a text input field
user_input = st.text_input("Enter your message massiii:")

# Display the input
if user_input:
    st.write(f"You entered: {user_input}")

# Button to send the input to the API
if st.button("Get Prediction"):
    if input_text:
        # Call the FastAPI backend
        try:
            response = requests.post("http://api:8000/predict", json={"text": input_text})
            if response.status_code == 200:
                result = response.json()
                st.success(f"Model Output: {result['output']}")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to the API: {e}")
    else:
        st.warning("Please enter some text!")

# Health check (optional)
if st.button("Check API Health"):
    try:
        response = requests.get("http://api:8000/health")
        if response.status_code == 200:
            st.success(f"API Health: {response.json()['status']}")
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to the API: {e}")
