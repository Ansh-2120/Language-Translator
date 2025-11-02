import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Dot, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate

data_path = r"D:\Coding\Language Translator\padded_sequences.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

X_en = data["X_en"]
Y_fr_in = data["Y_fr_in"]
Y_fr_out = data["Y_fr_out"]
vocab_size_en = data["vocab_size_en"]
vocab_size_fr = data["vocab_size_fr"]
max_len_en = data["max_len_en"]
max_len_fr = data["max_len_fr"]

print("✅ Loaded padded data")
print(f"Shapes: X_en={X_en.shape}, Y_fr_in={Y_fr_in.shape}, Y_fr_out={Y_fr_out.shape}")

embedding_dim = 128
lstm_units = 256

def attention_layer(decoder_outputs, encoder_outputs):
    attention_scores = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention_weights = Activation('softmax')(attention_scores)
    context_vector = Dot(axes=[2, 1])([attention_weights, encoder_outputs])
    return context_vector

# Encoder
encoder_inputs = Input(shape=(max_len_en,))
enc_emb = Embedding(vocab_size_en, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True, return_sequences=True)(enc_emb)

# Decoder
decoder_inputs = Input(shape=(max_len_fr,))
dec_emb = Embedding(vocab_size_fr, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

context_vector = attention_layer(decoder_outputs, encoder_outputs)
decoder_concat = Concatenate(axis=-1)([decoder_outputs, context_vector])
decoder_dense = Dense(vocab_size_fr, activation='softmax')
outputs = decoder_dense(decoder_concat)


model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


history = model.fit(
    [X_en, Y_fr_in], Y_fr_out,
    batch_size=32,
    epochs=15,
    validation_split=0.1
)
model.save(r"D:\Coding\Language Translator\nmt_model.h5")
print("💾 Model training complete and saved successfully!")