from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define a request model
class InputData(BaseModel):
    text: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Dummy prediction logic
    input_text = data.text
    response = f"Received: {input_text}. Here's a dummy response!"
    return {"input": input_text, "output": response}
