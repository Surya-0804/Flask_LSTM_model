import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# Hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SEQUENCE_LENGTH = 60  # Length of input sequences


import os
from collections import Counter

# Load and preprocess the text data directly from a file path
def load_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return text

def preprocess_text(text):
    tokens = text.split()
    vocab = Counter(tokens)
    vocab = sorted(vocab, key=vocab.get, reverse=True)
    vocab_to_int = {word: idx for idx, word in enumerate(vocab, 1)}
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}
    encoded_text = [vocab_to_int[word] for word in tokens]

    # Create input-output pairs
    sequences = []
    for i in range(0, len(encoded_text) - SEQUENCE_LENGTH):
        input_seq = encoded_text[i:i + SEQUENCE_LENGTH]
        target_seq = encoded_text[i + 1:i + SEQUENCE_LENGTH + 1]
        sequences.append((input_seq, target_seq))

    return sequences, vocab_to_int, int_to_vocab

# Directly provide the file path
file_path = r'C:\Users\hp\Documents\pythonlearn.txt'
text = load_text(file_path)
sequences, vocab_to_int, int_to_vocab = preprocess_text(text)
vocab_size = len(vocab_to_int) + 1
print(f'Vocabulary Size: {vocab_size}')

print(f'Vocabulary Size: {vocab_size}')



# Define the LSTM-based model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, sequences, vocab_size, epochs, batch_size, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)  # Move model to GPU

    # Split data into training and validation sets
    train_sequences, val_sequences = train_test_split(sequences, test_size=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        random.shuffle(train_sequences)
        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i + batch_size]
            inputs = torch.tensor([seq[0] for seq in batch_sequences], dtype=torch.long).to(device)
            targets = torch.tensor([seq[1] for seq in batch_sequences], dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss/len(train_sequences):.4f}')

# Initialize model
model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

# Train the model
train_model(model, sequences, vocab_size, EPOCHS, BATCH_SIZE, LEARNING_RATE)


def preprocess_text(text):
    tokens = text.split()
    vocab = Counter(tokens)
    vocab = sorted(vocab, key=vocab.get, reverse=True)
    vocab_to_int = {word: idx for idx, word in enumerate(vocab, 1)}
    vocab_to_int['<unk>'] = 0  # Add unknown token to vocabulary
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}
    encoded_text = [vocab_to_int.get(word, vocab_to_int['<unk>']) for word in tokens]  # Use <unk> for unknown words

    # Create input-output pairs
    sequences = []
    for i in range(0, len(encoded_text) - SEQUENCE_LENGTH):
        input_seq = encoded_text[i:i + SEQUENCE_LENGTH]
        target_seq = encoded_text[i + 1:i + SEQUENCE_LENGTH + 1]
        sequences.append((input_seq, target_seq))

    # print("Vocabulary to Int Mapping:", vocab_to_int)  # Debugging line
    # print("Int to Vocabulary Mapping:", int_to_vocab)   # Debugging line

    return sequences, vocab_to_int, int_to_vocab

def generate_text(model, start_text, int_to_vocab, vocab_to_int, length=100):
    model.eval()

    # Verify if '<unk>' token exists in vocab_to_int
    if '<unk>' not in vocab_to_int:
        raise ValueError("Unknown token '<unk>' is not in the vocabulary.")

    input_seq = [vocab_to_int.get(word, vocab_to_int['<unk>']) for word in start_text.split()]  # Handle unknown words
    generated_text = start_text

    for _ in range(length):
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(input_tensor)

        next_word_idx = torch.argmax(output[:, -1, :], dim=1).item()
        print(f"Predicted index: {next_word_idx,int_to_vocab[next_word_idx]}")  # Debugging line

        # Check if next_word_idx is within range of int_to_vocab
        
        next_word = int_to_vocab[next_word_idx]
        print(next_word)

        generated_text += ' ' + next_word
        input_seq.append(next_word_idx)
        input_seq = input_seq[1:]  # Slide the window

    return generated_text

# Load and preprocess the text
text = load_text(file_path)
sequences, vocab_to_int, int_to_vocab = preprocess_text(text)
vocab_size = len(vocab_to_int) + 1
print(f'Vocabulary Size: {vocab_size}')

# Define the LSTM-based model
# model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

# # Train the model
# train_model(model, sequences, vocab_size, EPOCHS, BATCH_SIZE, LEARNING_RATE)

# Generate text
start_text = "Writing programs"  # Replace with your input sentence
try:
    generated_text = generate_text(model, start_text, int_to_vocab, vocab_to_int)
    print("Generated Text: ", generated_text)
except Exception as e:
    print("Error during text generation:", e)
