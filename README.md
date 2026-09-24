# 📩 SMS Spam Classifier (Machine Learning + Streamlit)

Ek simple ML project jo SMS message ko **Spam** ya **Ham (Not Spam)** classify karta hai.

## Files
- `spam.csv` → Dataset (UCI SMS Spam Collection, 5572 messages)
- `train_model.py` → Data clean karke TF-IDF + Naive Bayes model train karta hai
- `app.py` → Streamlit web app (prediction ke liye)
- `requirements.txt` → Required Python packages

## Kaise chalayein (VS Code mein)

1. Folder ko VS Code mein open karo, terminal kholo.
2. Virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Requirements install karo:
   ```bash
   pip install -r requirements.txt
   ```
4. Model train karo (isse `model.pkl` aur `vectorizer.pkl` ban jayenge):
   ```bash
   python train_model.py
   ```
5. Streamlit app chalao:
   ```bash
   streamlit run app.py
   ```
6. Browser mein automatically khul jayega (`http://localhost:8501`). Wahan koi bhi SMS likho aur "Predict" dabao.

## Kaam kaise karta hai (short explanation for report/viva)
1. **Data Cleaning**: Text lowercase, URLs/numbers/punctuation remove, stopwords hataye.
2. **Feature Extraction**: TF-IDF vectorizer (top 3000 words) se text ko numbers mein convert kiya.
3. **Model**: Multinomial Naive Bayes — text classification ke liye classic aur fast algorithm.
4. **Result**: ~97% test accuracy.

## Improvements (agar aur marks chahiye)
- Naive Bayes ki jagah Logistic Regression / SVM try karke compare kar sakte ho.
- Confusion matrix aur ROC curve ka graph app mein dikha sakte ho.
- Word cloud (spam vs ham words) add kar sakte ho for EDA section.
