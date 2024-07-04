from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

def extract_features_tfidf(df):
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(df['preprocessed_text'])
    return X_tfidf

def extract_features_count(df):
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(df['preprocessed_text'])
    return X_count

def extract_features_lsa(X_tfidf):
    lsa = TruncatedSVD(n_components=2)
    X_lsa = lsa.fit_transform(X_tfidf)
    return X_lsa

if __name__ == "__main__":
    from read_data import read_data
    from preprocess_data import preprocess_dataframe

    df = read_data('text_video.xlsx')
    df = preprocess_dataframe(df)

    X_tfidf = extract_features_tfidf(df)
    X_count = extract_features_count(df)
    X_lsa = extract_features_lsa(X_tfidf)

    print(X_tfidf.shape, X_count.shape, X_lsa.shape)
