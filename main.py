"""FastAPI app that serves the churn model saved by discovery.py.

Run with:  uv run fastapi dev main.py
Then open http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pipeline once when the server starts, not on every request
    models["churn"] = joblib.load("model.joblib")
    yield
    models.clear()


app = FastAPI(title="Telco Churn Predictor", lifespan=lifespan)

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 3,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 95.5,
                    "TotalCharges": 286.5,
                }
            ]
        }
    }


@app.get("/")
def home():
    return {"message": "Churn model is running. Go to /docs to try it."}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    model = models["churn"]
    prediction = model.predict(row)[0]
    churn_probability = model.predict_proba(row)[0][list(model.classes_).index("Yes")]
    return {
        "churn": prediction,
        "churn_probability": round(float(churn_probability), 3),
    }
