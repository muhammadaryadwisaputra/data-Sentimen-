import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import pandas as pd

# Unduh data NLTK yang diperlukan
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    if pd.isnull(text):  # Periksa apakah teks adalah NaN
        return ''
    text = str(text)  # Ubah ke string untuk memastikan
    text = text.lower()  # Huruf kecil
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)  # Hapus angka
    words = word_tokenize(text)  # Tokenisasi
    stop_words = set(stopwords.words('indonesian'))
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Stemming dan hapus stopwords
    return " ".join(words)

def preprocess_dataframe(df):
    df['preprocessed_text'] = df['text'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    from read_data import read_data
    
    df = read_data('text_video.xlsx')  # Ganti ini sesuai dengan fungsi baca data Anda
    df = preprocess_dataframe(df)
    
    # Atur jumlah baris yang ingin ditampilkan
    pd.set_option('display.max_rows', 20)
    
    # Tampilkan 15 baris pertama
    print(df.head(20))
