from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
from groq import Groq

app = FastAPI(title="Milk Quality Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load trained model and label encoder
with open("milk_model.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

# Initialize Groq client
client = Groq(api_key="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")

class MilkData(BaseModel):
    pH: float = Field(..., ge=3.0, le=9.5, description="pH value between 3.0 and 9.5")
    Temprature: float = Field(..., ge=34.0, le=90.0, description="Temperature in Celsius between 34°C and 90°C")
    Odor: int = Field(..., ge=0, le=1, description="Methane detected (1=Present, 0=Not present)")
    Colour: int = Field(..., ge=240, le=255, description="Color value in RGB scale (240-255)")

def get_explanation(prediction: str, data: MilkData) -> str:
    prompt = (
        f"Milk quality is predicted as '{prediction}'. Explain why based on the given attributes:\n"
        f"- pH: {data.pH} (normal range: 6.25-6.90)\n"
        f"- Temperature: {data.Temprature}°C (normal range: 34°C-45.20°C)\n"
        f"- Methane odor: {'Present' if data.Odor == 1 else 'Not present'} (methane indicates bacterial activity)\n"
        f"- Colour: {data.Colour} on RGB scale (whiter milk with values closer to 255 typically indicates higher quality)\n"
        "If the quality is low or medium, suggest alternative uses (e.g., making cheese, baking, etc.)."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=1
    )

    return response.choices[0].message.content.strip()

@app.post("/predict", response_model=dict)
def predict_milk_quality(data: MilkData):
    try:
        # Create input array with only the kept features
        input_data = np.array([[data.pH, data.Temprature, data.Odor, data.Colour]])
        
        # Make prediction - get numeric prediction first
        numeric_prediction = model.predict(input_data)[0]
        
        # Convert numeric prediction back to original label
        prediction = label_encoder.inverse_transform([numeric_prediction])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        class_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Get explanation from LLM
        explanation = get_explanation(prediction, data)
        
        return {
            "quality": prediction, 
            "confidence": class_probabilities,
            "explanation": explanation,
            "input_values": {
                "pH": data.pH,
                "Temperature": data.Temprature,
                "Methane_Odor": "Present" if data.Odor == 1 else "Not present",
                "Colour": data.Colour
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Milk Quality Prediction API. Use /predict endpoint with POST request."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)