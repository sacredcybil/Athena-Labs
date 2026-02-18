
import pandas as pd
import pickle


with open("model/insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)




def get_recommendation(age, marital_status, has_kids, income_bracket, life_event):
    """
    Takes a new customer's information and returns ranked insurance recommendations.

    Parameters:
        age            (int)  : customer's age, e.g. 34
        marital_status (str)  : "single", "married", "divorced", or "widowed"
        has_kids       (int)  : 1 if they have children, 0 if not
        income_bracket (str)  : "low", "medium", or "high"
        life_event     (str)  : "none", "new_baby", "new_job", "retirement",
                                "marriage", "divorce", or "bought_home"

    Returns:
        list of dicts: ranked insurance recommendations with confidence scores
    """

    # Encode the text fields into numbers using the saved encoders
    encoded_marital  = encoders["marital_status"].transform([marital_status])[0]
    encoded_income   = encoders["income_bracket"].transform([income_bracket])[0]
    encoded_event    = encoders["life_event"].transform([life_event])[0]

    #one-row dataframe for a customer
    customer = pd.DataFrame([{
        "age":            age,
        "marital_status": encoded_marital,
        "has_kids":       has_kids,
        "income_bracket": encoded_income,
        "life_event":     encoded_event
    }])

    # Get probability scores for each insurance type
    probs = model.predict_proba(customer)[0]

    # Get the insurance product names in the same order as the probabilities
    insurance_labels = encoders["recommended_insurance"].classes_

    # Combine labels with their scores and sort by confidence (highest first)
    results = sorted(
        [{"insurance": label, "confidence": round(float(prob), 3)}
         for label, prob in zip(insurance_labels, probs)],
        key=lambda x: x["confidence"],
        reverse=True
    )

    return results


# --- EXAMPLE: RUN A TEST CUSTOMER ---
if __name__ == "__main__":
    # Edit these values to test different customer profiles
    test_customer = {
        "age":            34,
        "marital_status": "married",
        "has_kids":       1,
        "income_bracket": "medium",
        "life_event":     "new_baby"
    }

    print("\n Customer Profile:")
    for key, val in test_customer.items():
        print(f"   {key}: {val}")

    recommendations = get_recommendation(**test_customer)

    print("\n Recommendations (ranked by confidence):")
    for i, rec in enumerate(recommendations, 1):
        bar = "█" * int(rec["confidence"] * 20)
        print(f"   {i}. {rec['insurance']:<15} {rec['confidence']:.1%}  {bar}")
