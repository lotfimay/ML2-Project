from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define a request model
class InputData(BaseModel):
    text: str
    user_input: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    # Exemple de traitement
    processed_text = input_data.text.upper()  # Placeholder pour votre logique
    return {"result": processed_text, "user_input": input_data.user_input}
