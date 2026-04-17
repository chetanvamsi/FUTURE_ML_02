import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# load data
data = pd.read_csv("tickets.csv")

# convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Text'])

y = data['Category']

# train model
model = MultinomialNB()
model.fit(X, y)

# test
test = ["payment issue"]
test_vec = vectorizer.transform(test)

print("Category:", model.predict(test_vec))

def priority(text):
    if "urgent" in text:
        return "High"
    elif "issue" in text:
        return "Medium"
    else:
        return "Low"

print("Priority:", priority("urgent payment issue"))