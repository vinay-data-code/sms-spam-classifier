"""
SMS Spam Classifier - Training Script
--------------------------------------
Yeh script spam.csv dataset ko padhta hai, text ko clean karta hai,
TF-IDF + Naive Bayes model train karta hai, aur model.pkl / vectorizer.pkl
save karta hai jo app.py (Streamlit) use karega.

Run karne ke liye:
    python train_model.py
"""

import re
import string
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

# Common English stopwords (kept local so we don't depend on nltk downloads)
STOPWORDS = set("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down
in out on off over under again further then once here there when where why
how all any both each few more most other some such no nor not only own same
so than too very s t can will just don don't should now
""".split())


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers/extra spaces, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # urls
    text = re.sub(r"\d+", " ", text)                       # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


def main():
    # 1. Load data (dataset has some junk unnamed columns, encoding is latin-1)
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df.drop_duplicates(inplace=True)

    print("Total messages:", len(df))
    print(df["label"].value_counts())

    # 2. Clean text
    df["clean_message"] = df["message"].apply(clean_text)

    # 3. Encode labels: ham=0, spam=1
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    # 5. TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 6. Train model (Multinomial Naive Bayes -> classic, fast, works great for text)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    print("\n--- Model Performance ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 8. Save model + vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nSaved model.pkl and vectorizer.pkl. Ab app.py chala sakte ho:  streamlit run app.py")


if __name__ == "__main__":
    main()
