import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class CustomerData(BaseModel):
    age: int
    logins_per_month: int
    support_tickets: int
    plan_price: int
    months_subscribed: int

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/predict")
def predict(customer: CustomerData):
    features = np.array([[
        customer.age,
        customer.logins_per_month,
        customer.support_tickets,
        customer.plan_price,
        customer.months_subscribed
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "churned": bool(prediction),
        "churn_probability": round(float(probability), 3),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
def metrics():
    return {"model": "random_forest", "version": "1.0.0", "status": "serving"}