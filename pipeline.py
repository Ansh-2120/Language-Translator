import numpy as np
import tensorflow as tf
import pickle

with open(r"D:\Coding\Language Translator\tokenizer_en.pkl", "rb") as f:
    tokenizer_en = pickle.load(f)
with open(r"D:\Coding\Language Translator\tokenizer_fr.pkl", "rb") as f:
    tokenizer_fr = pickle.load(f)

model = tf.keras.models.load_model(r"D:\Coding\Language Translator\nmt_model.h5", compile=False)

reverse_fr_index = {v: k for k, v in tokenizer_fr.word_index.items()}
max_len_en = 76  
max_len_fr = 84

def translate_sentence(sentence):
    sentence = sentence.lower().strip()
    seq = tokenizer_en.texts_to_sequences([sentence])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len_en, padding='post')

    start_token = tokenizer_fr.word_index['<sos>']
    end_token = tokenizer_fr.word_index['<eos>']

    output_seq = [start_token]
    for _ in range(max_len_fr):
        preds = model.predict([seq, np.array([output_seq])], verbose=0)
        next_token = np.argmax(preds[0, -1, :])

        if next_token == end_token:
            break

        output_seq.append(next_token)

    translated_sentence = ' '.join([reverse_fr_index.get(t, '') for t in output_seq[1:]])
    return translated_sentence


if __name__ == "__main__":
    print("\n🌍 English → French Translator (Type 'exit' to quit)\n")

    while True:
        sentence = input("🟦 Enter English sentence: ").strip()
        if sentence.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            translation = translate_sentence(sentence)
            print(f"🟨 French: {translation}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
