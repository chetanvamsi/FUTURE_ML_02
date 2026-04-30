# ===============================
# STEP 1: Import Libraries
# ===============================
import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



# ===============================
# STEP 2: Create Sample Dataset
# (You can replace this with your CSV later)
# ===============================
data = pd.DataFrame({
    "ticket_text": [
        "Payment failed", "Refund not received", "Billing issue", "Incorrect charge",
        "App not working", "Login failed", "System crash", "Page not loading",
        "General inquiry", "Need help", "Information request", "How to use app",
        "Transaction failed", "Payment declined", "Unable to pay bill",
        "App is slow", "App crashes frequently", "Error message showing",
        "Account help needed", "Support required"
    ],
    "category": [
        "Billing","Billing","Billing","Billing",
        "Technical","Technical","Technical","Technical",
        "General","General","General","General",
        "Billing","Billing","Billing",
        "Technical","Technical","Technical",
        "General","General"
    ]
})
# ===============================
# STEP 3: Text Cleaning Function
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

data['cleaned_text'] = data['ticket_text'].apply(clean_text)

# ===============================
# STEP 4: Convert Text → Numbers (TF-IDF)
# ===============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])

# ===============================
# STEP 5: Encode Labels
# ===============================
le = LabelEncoder()
y = le.fit_transform(data['category'])

# ===============================
# STEP 6: Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ===============================
# STEP 7: Train Model (UPDATED)
# ===============================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# ===============================
# STEP 8: Evaluate Model
# ===============================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    zero_division=0
))
# ===============================
# STEP 9: Priority Function
# ===============================
def get_priority(text):
    text = text.lower()
    if "failed" in text or "not working" in text or "unable" in text:
        return "High"
    elif "slow" in text:
        return "Medium"
    else:
        return "Low"

# ===============================
# STEP 10: Test with New Input
# ===============================
def predict_ticket(ticket):
    clean = clean_text(ticket)
    vector = vectorizer.transform([clean])
    category_pred = model.predict(vector)
    category = le.inverse_transform(category_pred)[0]
    priority = get_priority(ticket)

    print("\n--- Prediction Result ---")
    print("Ticket:", ticket)
    print("Category:", category)
    print("Priority:", priority)

# Example test
predict_ticket("Payment not going through")
predict_ticket("App is very slow today")
predict_ticket("Unable to login to my account")

# ===============================
# STEP 12: User Input (Interactive)
# ===============================
user_input = input("\nEnter your support ticket: ")
predict_ticket(user_input)

# ===============================
# STEP 11: Visualization
# ===============================
data['category'].value_counts().plot(kind='bar')
plt.title("Ticket Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()