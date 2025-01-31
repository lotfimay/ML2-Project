import streamlit as st
import requests
import os

# Set page title and layout
st.set_page_config(page_title="Legal AI Assistant", layout="centered")

# API URL configuration
API_URL = os.getenv("API_URL", "http://ml2-api-alb-755946312.eu-north-1.elb.amazonaws.com")

# Title
st.markdown("<h2 style='text-align: center;'>🤖Legal AI Assistant</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Posez une question juridique, et notre modèle d'IA vous répondra.</p>", unsafe_allow_html=True)

# Input field for the user's legal question
question = st.text_area("🔍 Entrez votre question juridique :", placeholder="Exemple: Quels sont mes droits en cas de licenciement abusif?")

# Button to submit the question
if st.button("Obtenir une réponse"):
    if question.strip():
        # Prepare request payload
        payload = {
            "text": question, 
        }

        with st.spinner("Analyse de votre question... ⏳"):
            try:
                # Send request to API
                response = requests.post(f"{API_URL}/predict", json=payload)

                # Process response
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Réponse obtenue avec succès !")
                    st.markdown(f"**📌 Question :** {question}")
                    st.markdown(f"**📝 Réponse :** {result['result']}")

                else:
                    st.error(f"🚨 Erreur API : {response.status_code}")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Problème de connexion avec l'API : {e}")

    else:
        st.warning("⚠️ Veuillez entrer une question avant de soumettre.")

# Button to check API status
if st.button("📡 Vérifier l'état de l'API"):
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            st.success(f"🟢 API en ligne : {response.json()['status']}")
        else:
            st.error(f"🔴 Problème avec l'API : {response.status_code}")
    except Exception as e:
        st.error(f"❌ Impossible de contacter l'API : {e}")