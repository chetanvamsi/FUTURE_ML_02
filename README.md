# Support Ticket Classification (Task 2)

## 📌 Objective

The goal of this project is to build a Machine Learning model that can automatically classify customer support tickets into categories and assign priority levels.

---

## 🛠️ Tools & Technologies Used

* Python
* Pandas
* Scikit-learn
* CountVectorizer
* Multinomial Naive Bayes

---

## 📂 Dataset

A simple dataset (`tickets.csv`) was created with two columns:

* **Text** → Customer message
* **Category** → Type of issue (Billing / Technical)

Example:

```
payment failed → Billing
login issue → Technical
```

---

## ⚙️ Steps Performed

1. Loaded dataset using Pandas
2. Converted text data into numerical form using CountVectorizer
3. Trained a classification model using Naive Bayes
4. Predicted category for new support tickets
5. Added priority logic (High / Medium / Low)

---

## ▶️ How It Works

* Input: `"payment issue"`
* Output:

  * Category: Billing
  * Priority: Medium

---

## 📊 Output

The model successfully classifies support tickets and assigns priority based on keywords.

---

## 📁 Project Structure

```
FUTURE_ML_02/
│
├── classification.ipynb
├── tickets.csv
└── README.md
```

---

## 🚀 Conclusion

This project demonstrates how Machine Learning can automate customer support by quickly classifying tickets and prioritizing them, helping teams respond faster.

---
