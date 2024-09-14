import torch
import torch.nn as nn
from collections import Counter

# Define the LSTM-based model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Ensure vocab_size matches the output layer


    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

# Load and preprocess the text data directly from a file path
def preprocess_text(text, sequence_length):
    tokens = text.split()
    vocab = Counter(tokens)
    vocab = sorted(vocab, key=vocab.get, reverse=True)
    vocab_to_int = {word: idx for idx, word in enumerate(vocab, 1)}
    vocab_to_int['<unk>'] = 0  # Add unknown token to vocabulary
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}
    encoded_text = [vocab_to_int.get(word, vocab_to_int['<unk>']) for word in tokens]  # Use <unk> for unknown words

    # Create input-output pairs
    sequences = []
    for i in range(0, len(encoded_text) - sequence_length):
        input_seq = encoded_text[i:i + sequence_length]
        target_seq = encoded_text[i + 1:i + sequence_length + 1]
        sequences.append((input_seq, target_seq))

    print("Vocabulary to Int Mapping:", vocab_to_int)  # Debugging line
    print("Int to Vocabulary Mapping:", int_to_vocab)   # Debugging line

    return sequences, vocab_to_int, int_to_vocab
