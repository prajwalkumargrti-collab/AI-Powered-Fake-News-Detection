import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


nltk.download("stopwords")


true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true["label"] = 1      # Real News
fake["label"] = 0      # Fake News

data = pd.concat([true, fake], ignore_index=True)

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset Information")
print(data["label"].value_counts())
print(data.head())


data.drop(columns=["subject", "date"], inplace=True, errors="ignore")


data.fillna("", inplace=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["content"] = (data["title"] + " " + data["text"]).apply(clean_text)


tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X = tfidf.fit_transform(data["content"])
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42
    )
}

best_model = None
best_accuracy = 0
best_name = ""


for name, model in models.items():

    print("\n==============================")
    print("Training:", name)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Prediction Counts:")
    print(pd.Series(pred).value_counts())
    print("Actual Counts:")
    print(pd.Series(y_test).value_counts())

    accuracy = accuracy_score(y_test, pred)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print(classification_report(y_test, pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print("\n==============================")
print("Best Model:", best_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")
news = data.iloc[100]["content"]
print(news)

index = 100

sample = X_test[index]
actual = y_test.iloc[index]
predicted = best_model.predict(sample)

print("Actual:", actual)
print("Predicted:", predicted[0])
# -------------------------------
# Prediction
# -------------------------------
while True:

    news = input("\nEnter News (type 'exit' to quit): ")

    if news.lower() == "exit":
        print("Program Closed.")
        break

    news = clean_text(news)

    news_vector = tfidf.transform([news])

    prediction = best_model.predict(news_vector)
    probability = best_model.predict_proba(news_vector)

    print("\nRaw Prediction:", prediction)

    confidence = max(probability[0]) * 100

    if prediction[0] == 1:
        print("Prediction : REAL NEWS")
    else:
        print("Prediction : FAKE NEWS")

    print("Confidence :", round(confidence, 2), "%")