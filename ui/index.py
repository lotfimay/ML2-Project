import streamlit as st
import requests
import os

# Titre de l'application
st.title("NLP Model Tester")

# Boîte de saisie de texte
input_text = st.text_area("Entrez du texte :", "")

# Champ de saisie supplémentaire
user_input = st.text_input("Entrez votre message massiiinissa:")

# Bouton de soumission
if st.button("Soumettre"):
    if input_text:
        # Récupérer l'URL de l'API depuis les variables d'environnement
        API_URL = os.getenv("API_URL", "http://ml2-api-alb-755946312.eu-north-1.elb.amazonaws.com/api")
        
        # Préparer les données à envoyer
        payload = {
            "input_text": input_text,
            "user_input": user_input
        }

        try:
            # Effectuer une requête POST vers l'API
            response = requests.post(f"{API_URL}/process", json=payload)

            # Vérifier la réponse
            if response.status_code == 200:
                result = response.json()
                st.success("Traitement réussi!")
                st.write("Résultat :", result)
            else:
                st.error(f"Erreur de l'API : {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la connexion à l'API : {e}")
    else:
        st.warning("Veuillez entrer du texte avant de soumettre.")

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
