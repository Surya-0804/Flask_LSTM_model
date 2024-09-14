import torch
import torch.optim as optim
import random
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from model.LSTM_Model import LSTMModel, preprocess_text
import json

def train_model(model, sequences, vocab_size, epochs, batch_size, lr, device):
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

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

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_sequences):.4f}')

def main():
    # Define hyperparameters
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    EPOCHS = 2
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    SEQUENCE_LENGTH = 30  # Define sequence length
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess text
    file_path = r'C:\Users\hp\Documents\pythonlearn.txt'
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    sequences, vocab_to_int, int_to_vocab = preprocess_text(text, SEQUENCE_LENGTH)

    # Initialize model and train
    vocab_size = len(vocab_to_int) + 1
    model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    train_model(model, sequences, vocab_size, EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE)

    # Save vocab mappings and model
    with open('model/vocab_to_int.json', 'w') as f:
        json.dump(vocab_to_int, f)

    with open('model/int_to_vocab.json', 'w') as f:
        json.dump(int_to_vocab, f)

    torch.save(model.state_dict(), 'model/lstm_model.pth')

if __name__ == "__main__":
    main()
