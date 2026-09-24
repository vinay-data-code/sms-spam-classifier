# 📩 SMS Spam Classifier (Machine Learning + Streamlit)

A simple ML project that classifies an SMS message as **Spam** or **Ham (Not Spam)**.

## Files
- `spam.csv` → Dataset (UCI SMS Spam Collection, 5572 messages)
- `train_model.py` → Cleans the data and trains a TF-IDF + Naive Bayes model
- `app.py` → Streamlit web app (for prediction)
- `requirements.txt` → Required Python packages

## How to run (in VS Code)

1. Open the folder in VS Code and open a terminal.
2. Create a virtual environment (optional but recommended):
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
```
3. Install requirements:
```bash
   pip install -r requirements.txt
```
4. Train the model (this creates `model.pkl` and `vectorizer.pkl`):
```bash
   python train_model.py
```
5. Run the Streamlit app:
```bash
   streamlit run app.py
```
6. It will automatically open in your browser (`http://localhost:8501`). Type or paste any SMS and click "Predict".

## How it works (short explanation for report/viva)
1. **Data Cleaning**: Lowercase the text, remove URLs/numbers/punctuation, remove stopwords.
2. **Feature Extraction**: TF-IDF vectorizer (top 3000 words) converts text into numbers.
3. **Model**: Multinomial Naive Bayes — a classic, fast algorithm for text classification.
4. **Result**: ~97% test accuracy.

## Possible improvements (for extra marks)
- Try Logistic Regression / SVM instead of Naive Bayes and compare results.
- Show a confusion matrix and ROC curve graph in the app.
- Add a word cloud (spam vs ham words) for the EDA section.