from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def cluster_kmeans(X):
    kmeans = KMeans(n_clusters=2)
    y_kmeans = kmeans.fit_predict(X)
    print("K-Means Silhouette Score:", silhouette_score(X, y_kmeans))

def cluster_dbscan(X):
    dbscan = DBSCAN()
    y_dbscan = dbscan.fit_predict(X)
    print("DBSCAN Silhouette Score:", silhouette_score(X, y_dbscan))

if __name__ == "__main__":
    from read_data import read_data
    from preprocess_data import preprocess_dataframe
    from feature_extraction import extract_features_tfidf, extract_features_count, extract_features_lsa

    df = read_data('text_video.xlsx')
    df = preprocess_dataframe(df)

    X_tfidf = extract_features_tfidf(df)
    X_count = extract_features_count(df)
    X_lsa = extract_features_lsa(X_tfidf)

    print("TF-IDF:")
    cluster_kmeans(X_tfidf)
    cluster_dbscan(X_tfidf)

    print("Count Vectorizer:")
    cluster_kmeans(X_count)
    cluster_dbscan(X_count)

    print("LSA:")
    cluster_kmeans(X_lsa)
    cluster_dbscan(X_lsa)
