import pandas as pd

def read_data(file_path):
    df = pd.read_excel(file_path)
    return df

if __name__ == "__main__":
    file_path = 'text_video.xlsx'
    df = read_data(file_path)
    print(df.head())
