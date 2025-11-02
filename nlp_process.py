import pandas as pd
import spacy

df = pd.read_csv(r"D:\Coding\Language Translator\cleaned_en_fr.csv")
df['en'] = df['en'].fillna("").astype(str)
df['fr'] = df['fr'].fillna("").astype(str)

def is_valid_pair(en, fr, min_len=1, max_len=60, ratio_thresh=2.5):
    en_tokens = en.split()
    fr_tokens = fr.split()
    
    if len(en_tokens) < min_len or len(fr_tokens) < min_len:
        return False
    
    if len(en_tokens) > max_len or len(fr_tokens) > max_len:
        return False
    
    ratio = max(len(en_tokens), len(fr_tokens)) / max(len(en_tokens), 1)
    if ratio > ratio_thresh:
        return False
    
    return True

df = df[df.apply(lambda row: is_valid_pair(row['en'], row['fr']), axis=1)]
print(f"✅ After filtering: {len(df)} sentence pairs remain")

spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

df['en_tokens'] = df['en'].apply(tokenize_en)
df['fr_tokens'] = df['fr'].apply(tokenize_fr)

df['fr_in'] = df['fr_tokens'].apply(lambda x: ['<SOS>'] + x)
df['fr_out'] = df['fr_tokens'].apply(lambda x: x + ['<EOS>'])

save_path = r"D:\Coding\Language Translator\processed_en_fr_tokens.pkl"
df.to_pickle(save_path)

print(f"✅ Tokenized dataset saved to: {save_path}")