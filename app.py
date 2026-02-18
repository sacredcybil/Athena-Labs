"""
STEP 4: app.py
--------------
A simple command-line chatbot that:
1. Asks the user for their information
2. Passes it to the ML model to get recommendation scores
3. Passes those scores to a local LLM (via Ollama) to generate a friendly explanation

PREREQUISITES:
- Install Ollama from https://ollama.com
- Run: ollama pull llama3
- Then run this file: python app.py
"""

import requests
from recommend import get_recommendation

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  


def ask_llm(customer_info: dict, recommendations: list) -> str:
    """
    Sends the customer profile and ML model scores to the local LLM
    and gets back a friendly, personalized explanation.
    """

    # Format recommendations for the prompt
    rec_text = "\n".join(
        [f"  - {r['insurance']}: {r['confidence']:.0%} confidence" for r in recommendations]
    )

    prompt = f"""You are a friendly and knowledgeable insurance advisor.

A customer has shared the following information:
- Age: {customer_info['age']}
- Marital status: {customer_info['marital_status']}
- Has children: {'Yes' if customer_info['has_kids'] else 'No'}
- Income bracket: {customer_info['income_bracket']}
- Recent life event: {customer_info['life_event']}

Based on an analysis of their profile, here are the insurance recommendations ranked by suitability:
{rec_text}

Please write a warm, clear, 3-4 sentence explanation for why the top recommendation suits this customer.
Mention the second option briefly if it's also relevant. Do not use bullet points."""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=60)

        return response.json()["response"]

    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to Ollama. Make sure it's running (try: ollama serve).\n"
            f"Your top recommendation is: {recommendations[0]['insurance']} "
            f"({recommendations[0]['confidence']:.0%} confidence)"
        )


def collect_customer_info() -> dict:
    """Prompts the user for their information via the command line."""

    print("\n" + "="*50)
    print("  Insurance Recommendation Chatbot")
    print("="*50)
    print("Please answer a few questions to get your personalized recommendation.\n")

    # Age
    while True:
        try:
            age = int(input("What is your age? "))
            if 18 <= age <= 100:
                break
            print("Please enter an age between 18 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    # Marital status
    marital_options = ["single", "married", "divorced", "widowed"]
    print(f"\nMarital status options: {', '.join(marital_options)}")
    while True:
        marital_status = input("What is your marital status? ").strip().lower()
        if marital_status in marital_options:
            break
        print(f"Please choose from: {', '.join(marital_options)}")

    # Children
    while True:
        kids_input = input("\nDo you have children? (yes/no): ").strip().lower()
        if kids_input in ["yes", "no"]:
            has_kids = 1 if kids_input == "yes" else 0
            break
        print("Please answer 'yes' or 'no'.")

    # Income bracket
    income_options = ["low", "medium", "high"]
    print(f"\nIncome bracket options: {', '.join(income_options)}")
    while True:
        income_bracket = input("What is your income bracket? ").strip().lower()
        if income_bracket in income_options:
            break
        print(f"Please choose from: {', '.join(income_options)}")

    # Life event
    event_options = ["none", "new_baby", "new_job", "retirement", "marriage", "divorce", "bought_home"]
    print(f"\nLife event options: {', '.join(event_options)}")
    while True:
        life_event = input("Have you experienced a recent life event? ").strip().lower()
        if life_event in event_options:
            break
        print(f"Please choose from: {', '.join(event_options)}")

    return {
        "age":            age,
        "marital_status": marital_status,
        "has_kids":       has_kids,
        "income_bracket": income_bracket,
        "life_event":     life_event
    }


if __name__ == "__main__":
    # Collect customer info
    customer_info = collect_customer_info()

    # Get ML model recommendations
    print("\n⏳ Analyzing your profile...")
    recommendations = get_recommendation(**customer_info)

    # Show raw scores
    print("\n📊 Model scores:")
    for rec in recommendations:
        bar = "█" * int(rec["confidence"] * 20)
        print(f"   {rec['insurance']:<15} {rec['confidence']:.0%}  {bar}")

    # Get LLM explanation
    print("\n💬 Advisor recommendation:\n")
    explanation = ask_llm(customer_info, recommendations)
    print(explanation)
    print()
