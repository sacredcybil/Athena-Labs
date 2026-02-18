
import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

N = 1000  #no of customers

ages = np.random.randint(18, 70, N)
marital_statuses = np.random.choice(["single", "married", "divorced", "widowed"], N, p=[0.4, 0.4, 0.1, 0.1])
has_kids = np.random.choice([0, 1], N, p=[0.5, 0.5])
income_brackets = np.random.choice(["low", "medium", "high"], N, p=[0.3, 0.5, 0.2])
life_events = np.random.choice(["none", "new_baby", "new_job", "retirement", "marriage", "divorce", "bought_home"], N
)


#domain rules

def recommend_insurance(age, marital_status, has_kids, income_bracket, life_event):
    if life_event == "new_baby" or (has_kids and marital_status == "married"):
        return "term_life"
    elif age >= 55 or life_event == "retirement":
        return "whole_life"
    elif life_event == "bought_home":
        return "home"
    elif income_bracket == "low" or life_event == "new_job":
        return "health"
    elif marital_status in ["divorced", "widowed"]:
        return "term_life"
    elif age < 30 and marital_status == "single":
        return "health"
    else:
        return random.choice(["health", "term_life", "auto"])



recommendations = [
    recommend_insurance(ages[i], marital_statuses[i], has_kids[i], income_brackets[i], life_events[i])
    for i in range(N)
]


df = pd.DataFrame({
    "age":                    ages,
    "marital_status":         marital_statuses,
    "has_kids":               has_kids,
    "income_bracket":         income_brackets,
    "life_event":             life_events,
    "recommended_insurance":  recommendations   
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic_dataset.csv", index=False)

print(f"✅ Dataset created with {N} customers → data/synthetic_dataset.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nRecommendation breakdown:")
print(df["recommended_insurance"].value_counts())
