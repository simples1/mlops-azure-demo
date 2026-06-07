import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Create realistic fake customer data
data = {
    "age":              [34, 28, 45, 52, 23, 38, 41, 29],
    "logins_per_month": [12,  2, 20,  1, 15,  8,  3, 18],
    "support_tickets":  [ 3,  8,  1,  9,  2,  4,  7,  1],
    "plan_price":       [99, 49,199, 49, 99,149, 49,199],
    "months_subscribed":[ 8,  2, 24,  1, 12, 18,  3, 36],
    "churned":          [ 0,  1,  0,  1,  0,  0,  1,  0]
}

df = pd.DataFrame(data)

X = df.drop(columns=["churned"])
y = df["churned"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
