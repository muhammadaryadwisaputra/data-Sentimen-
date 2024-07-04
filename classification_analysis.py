from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

def classify_naive_bayes(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Naive Bayes Classification Report:")
    print(classification_report(y, y_pred, zero_division=1))

def classify_svm(X, y):
    model = SVC()
    model.fit(X, y)
    y_pred = model.predict(X)
    print("SVM Classification Report:")
    print(classification_report(y, y_pred, zero_division=1))

def classify_logistic_regression(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Logistic Regression Classification Report:")
    print(classification_report(y, y_pred, zero_division=1))

if __name__ == "__main__":
    from read_data import read_data
    from preprocess_data import preprocess_dataframe
    from feature_extraction import extract_features_tfidf, extract_features_count, extract_features_lsa

    df = read_data('text_video.xlsx')
    df = preprocess_dataframe(df)

    # Memastikan kolom label tidak memiliki nilai NaN
    df = df.dropna(subset=['label'])

    X_tfidf = extract_features_tfidf(df)
    X_count = extract_features_count(df)
    X_lsa = extract_features_lsa(X_tfidf)

    y = df['label']

    # Normalisasi data untuk SVM dan Logistic Regression
    scaler = StandardScaler(with_mean=False)  # Menonaktifkan with_mean untuk matriks sparse
    X_tfidf_scaled = scaler.fit_transform(X_tfidf)
    X_count_scaled = scaler.fit_transform(X_count)
    X_lsa_scaled = scaler.fit_transform(X_lsa)

    print("TF-IDF:")
    classify_naive_bayes(X_tfidf, y)
    classify_svm(X_tfidf_scaled, y)

    print("Count Vectorizer:")
    classify_naive_bayes(X_count, y)
    classify_svm(X_count_scaled, y)

    print("LSA:")
    classify_svm(X_lsa_scaled, y)
    classify_logistic_regression(X_lsa_scaled, y)
