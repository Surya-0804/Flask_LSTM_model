import torch

def generate_text(model, start_text, int_to_vocab, vocab_to_int, length=100, device='cpu'):
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