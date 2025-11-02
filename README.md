# English → French Language Translator 🌍

A neural machine translation system that translates English sentences to French using a Seq2Seq model with Attention mechanism, trained on 200,000+ parallel sentence pairs from Europarl and News Commentary datasets.

## 📌 What It Does

This translator converts English sentences into French using deep learning. It handles:
- **Simple sentences**: "Hello, how are you?" → "bonjour, comment allez-vous?"
- **Complex phrases**: Long sentences with proper grammar
- **Real-time translation**: Interactive command-line interface

## 🛠️ Tech Stack

**Frameworks & Libraries:**
- **TensorFlow/Keras** - Neural network framework
- **Spacy** - Tokenization (en_core_web_sm, fr_core_news_sm)
- **Pandas** - Data processing
- **NumPy** - Numerical operations
- **Scikit-learn** - Data shuffling

**Model Architecture:**
- **Type**: Sequence-to-Sequence with Attention
- **Encoder**: LSTM (256 units) with embeddings (128 dim)
- **Decoder**: LSTM (256 units) with attention mechanism
- **Attention**: Dot-product attention for context vectors
- **Vocab Size**: ~30K English, ~45K French tokens

## ⚙️ How It Works

### **Tech Flow**

```
Raw Datasets (Europarl + News Commentary)
    ↓
preprocess.py → Text cleaning & combining
    ↓
nlp_process.py → Tokenization with Spacy
    ↓
nlp_vocab.py → Build vocabularies & pad sequences
    ↓
model.py → Train Seq2Seq + Attention model
    ↓
pipeline.py → Interactive translation interface
```

### **1. Data Preprocessing** (`preprocess.py`)

**What happens:**
- Loads Europarl (2M+ pairs) and News Commentary (400K+ pairs) datasets
- Cleans text: lowercase, remove HTML tags, normalize Unicode
- Filters to 200K sentence pairs for training
- Shuffles and saves as CSV

**Key operations:**
```python
Text normalization → HTML removal → Lowercasing → Whitespace cleanup
```

### **2. Tokenization** (`nlp_process.py`)

**What happens:**
- Uses Spacy for English and French tokenization
- Filters sentences (1-60 tokens, length ratio < 2.5)
- Adds special tokens: `<SOS>` (start) and `<EOS>` (end)
- Saves tokenized data as pickle

**Token flow:**
```
English: "I love coding"
    ↓
Tokens: ['I', 'love', 'coding']

French: "J'aime coder"
    ↓
Input:  ['<SOS>', "J'", 'aime', 'coder']
Output: ["J'", 'aime', 'coder', '<EOS>']
```

### **3. Vocabulary Building** (`nlp_vocab.py`)

**What happens:**
- Creates word-to-index mappings using Keras Tokenizer
- Pads sequences to max length (76 for EN, 84 for FR)
- Saves tokenizers and padded sequences

**Output:**
- `X_en`: English sequences (input)
- `Y_fr_in`: French sequences with `<SOS>` (decoder input)
- `Y_fr_out`: French sequences with `<EOS>` (target)

### **4. Model Training** (`model.py`)

**Architecture:**
```
ENCODER:
Input (max_len_en) → Embedding(128) → LSTM(256) → [encoder_outputs, state_h, state_c]

DECODER:
Input (max_len_fr) → Embedding(128) → LSTM(256, initial_state=[h, c]) → decoder_outputs
                                           ↓
                              ATTENTION MECHANISM
                        (Dot product with encoder_outputs)
                                           ↓
                              Context Vector + decoder_outputs
                                           ↓
                              Dense(vocab_size_fr, softmax) → French word probabilities
```

**Training config:**
- Optimizer: Adam
- Loss: Sparse categorical crossentropy
- Batch size: 32
- Epochs: 15
- Validation split: 10%

**Attention mechanism:**
- Calculates similarity between decoder and encoder outputs
- Creates context vector focusing on relevant source words
- Improves translation quality for long sentences

### **5. Translation Pipeline** (`pipeline.py`)

**Inference process:**
1. Tokenize English input
2. Encode with trained encoder
3. Decode word-by-word:
   - Start with `<SOS>` token
   - Predict next word using decoder + attention
   - Stop at `<EOS>` or max length
4. Convert token IDs back to French words

**Usage:**
```bash
python pipeline.py
```

## 📊 Dataset

**Sources:**
- **Europarl v10**: European Parliament proceedings (~2M pairs)
- **News Commentary v18**: News articles (~400K pairs)

**Processing:**
- Combined: 2.4M+ sentence pairs
- Filtered to: 200K pairs (for faster training)
- Max sentence length: 60 tokens
- Length ratio threshold: 2.5x

## 🚀 Training Process

1. **Load & Clean Data** - Normalize, filter, shuffle
2. **Tokenize** - Spacy tokenization for EN/FR
3. **Build Vocabularies** - Word-to-index mappings
4. **Pad Sequences** - Fixed-length inputs for neural network
5. **Train Seq2Seq** - 15 epochs with attention mechanism
6. **Save Model** - Export as `nmt_model.h5`
7. **Interactive Translation** - Real-time CLI interface

## 📁 Project Structure

```
├── preprocess.py              # Data loading and cleaning
├── nlp_process.py             # Spacy tokenization
├── nlp_vocab.py               # Vocabulary building & padding
├── model.py                   # Seq2Seq + Attention training
├── pipeline.py                # Translation inference
├── cleaned_en_fr.csv          # Preprocessed dataset
├── processed_en_fr_tokens.pkl # Tokenized data
├── tokenizer_en.pkl           # English vocabulary
├── tokenizer_fr.pkl           # French vocabulary
├── padded_sequences.pkl       # Padded input sequences
└── nmt_model.h5               # Trained model
```

## 🎯 Model Details

**Parameters:**
- Embedding dimension: 128
- LSTM units: 256
- Encoder vocab: ~30,000 words
- Decoder vocab: ~45,000 words
- Max input length: 76 tokens
- Max output length: 84 tokens

**Architecture highlights:**
- Bidirectional information flow via attention
- Teacher forcing during training
- Autoregressive decoding during inference

## 💻 Requirements

```
tensorflow>=2.x
spacy>=3.0
pandas
numpy
scikit-learn
```

**Spacy models:**
```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## 🔮 Future Improvements

- Add Transformer architecture (better than LSTM)
- Increase dataset size for better accuracy
- Implement beam search for better translations
- Add BLEU score evaluation
- Create web interface with Flask/FastAPI
- Support more language pairs

---

**Built with Seq2Seq + Attention to bridge language barriers! 🗣️✨**
