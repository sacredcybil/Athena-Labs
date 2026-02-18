
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb


df = pd.read_csv("data/synthetic_dataset.csv")


text_columns = ["marital_status", "income_bracket", "life_event", "recommended_insurance"]
encoders = {}

for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"   '{col}' categories: {list(le.classes_)}")


# X = the customer's information (inputs)
# y = the correct insurance recommendation 
X = df[["age", "marital_status", "has_kids", "income_bracket", "life_event"]]
y = df["recommended_insurance"]

#training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n Training on {len(X_train)} rows, testing on {len(X_test)} rows")


model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="mlogloss")
model.fit(X_train, y_train)
print("Model trained")

#evaluation
y_pred = model.predict(X_test)
insurance_labels = list(encoders["recommended_insurance"].classes_)
print("\n Model Performance:")
print(classification_report(y_test, y_pred, target_names=insurance_labels))

#to memory
os.makedirs("model", exist_ok=True)

with open("model/insurance_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Model saved → model/insurance_model.pkl")
print("Encoders saved → model/label_encoders.pkl")
