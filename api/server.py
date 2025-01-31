from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

from RAG import full_rag_pipeline

# Define a request model
class InputData(BaseModel):
    text: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):

    print("Heere")
    print("Predict end point called")
    # Exemple de traitement
    response = full_rag_pipeline(input_data.text)  # Placeholder pour votre logique
    print(response)
    return {"result": response}
