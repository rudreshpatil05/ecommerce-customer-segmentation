import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv("data/data.csv") 
# Create labels
def label(row):
    if row["Total_Orders"] > 20:
        return "Regular"
    elif row["Total_Orders"] > 5:
        return "Average"
    else:
        return "Low"

df["Category"] = df.apply(label, axis=1)

X = df[["Total_Orders", "Total_Spending", "Last_Login_Days"]]
y = df["Category"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
import pickle
with open("../models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")