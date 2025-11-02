import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenized_path = r"D:\Coding\Language Translator\processed_en_fr_tokens.pkl"
df = pd.read_pickle(tokenized_path)

print(f"✅ Loaded tokenized dataset with {len(df)} pairs")

en_texts = df['en_tokens'].apply(lambda x: ' '.join(x)).tolist()
fr_in_texts = df['fr_in'].apply(lambda x: ' '.join(x)).tolist()
fr_out_texts = df['fr_out'].apply(lambda x: ' '.join(x)).tolist()

tokenizer_en = Tokenizer(oov_token="<UNK>")
tokenizer_fr = Tokenizer(oov_token="<UNK>")
tokenizer_en.fit_on_texts(en_texts)
tokenizer_fr.fit_on_texts(fr_in_texts)

vocab_size_en = len(tokenizer_en.word_index) + 1
vocab_size_fr = len(tokenizer_fr.word_index) + 1

print(f"📘 English vocab size: {vocab_size_en}")
print(f"📗 French vocab size: {vocab_size_fr}")

X_en = tokenizer_en.texts_to_sequences(en_texts)
Y_fr_in = tokenizer_fr.texts_to_sequences(fr_in_texts)
Y_fr_out = tokenizer_fr.texts_to_sequences(fr_out_texts)

max_len_en = max(len(seq) for seq in X_en)
max_len_fr = max(len(seq) for seq in Y_fr_in)

X_en = pad_sequences(X_en, maxlen=max_len_en, padding='post')
Y_fr_in = pad_sequences(Y_fr_in, maxlen=max_len_fr, padding='post')
Y_fr_out = pad_sequences(Y_fr_out, maxlen=max_len_fr, padding='post')

print(f"✅ Shapes:")
print(f"   X_en:     {X_en.shape}")
print(f"   Y_fr_in:  {Y_fr_in.shape}")
print(f"   Y_fr_out: {Y_fr_out.shape}")

output_dir = r"D:\Coding\Language Translator"
with open(f"{output_dir}\\tokenizer_en.pkl", "wb") as f:
    pickle.dump(tokenizer_en, f)
with open(f"{output_dir}\\tokenizer_fr.pkl", "wb") as f:
    pickle.dump(tokenizer_fr, f)

pickle.dump({
    "X_en": X_en,
    "Y_fr_in": Y_fr_in,
    "Y_fr_out": Y_fr_out,
    "max_len_en": max_len_en,
    "max_len_fr": max_len_fr,
    "vocab_size_en": vocab_size_en,
    "vocab_size_fr": vocab_size_fr
}, open(f"{output_dir}\\padded_sequences.pkl", "wb"))

print("💾 Saved tokenizers and padded sequences successfully!")