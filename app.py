from flask import Flask, request, jsonify
import torch
import json
from model.LSTM_Model import LSTMModel
from model.generate_text import generate_text

app = Flask(__name__)

# Load int_to_vocab mapping
with open('model/vocab_to_int.json', 'r') as f:
    vocab_to_int = json.load(f)
vocab_size = len(vocab_to_int) + 1

# Load model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LSTMModel(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128)
model.load_state_dict(torch.load('model/lstm_model.pth', map_location=DEVICE,weights_only=True))
model.to(DEVICE)
print(DEVICE)
print(torch.cuda.get_device_name(0))  # Print the CUDA device name
print(torch.__version__)  # Print the PyTorch version
print(torch.version.cuda)  # Print the CUDA version



with open('model/int_to_vocab.json', 'r') as f:
    int_to_vocab = json.load(f)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    start_text = data['start_text']
    print(start_text)
    try:
        generated_text = generate_text(model, start_text, int_to_vocab, vocab_to_int, length=100, device=DEVICE)
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
