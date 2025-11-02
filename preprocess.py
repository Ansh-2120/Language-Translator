import pandas as pd
import re
import unicodedata
from sklearn.utils import shuffle

europarl_path = r"D:\Coding\Language Translator\europarl-v10.fr-en.pair.tsv"
news_path = r"D:\Coding\Language Translator\news-commentary-v18.en-fr.tsv"

def load_tsv(path, reverse=False):
    df = pd.read_csv(path, sep="\t", header=None, names=["col1", "col2"], quoting=3, on_bad_lines="skip")
    if reverse:
        df = df[["col2", "col1"]]
    df.columns = ["en", "fr"]
    return df

df_euro = load_tsv(europarl_path, reverse=True)
df_news = load_tsv(news_path, reverse=False)

df = pd.concat([df_euro, df_news], ignore_index=True)
print(f"✅ Combined dataset shape: {df.shape}")

def clean_text(text):
    if not isinstance(text, str): 
        return ""
    text = unicodedata.normalize('NFD', str(text))
    text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df = df.dropna(subset=["en", "fr"])
df['en'] = df['en'].apply(clean_text)
df['fr'] = df['fr'].apply(clean_text)

df = shuffle(df, random_state=42).reset_index(drop=True)

subset_size = 200_000
if len(df) > subset_size:
    df = df.sample(n=subset_size, random_state=42).reset_index(drop=True)

output_path = r"D:\Coding\Language Translator\cleaned_en_fr.csv"
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"✅ Preprocessing complete! Saved to {output_path}")