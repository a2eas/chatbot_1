# Transformer Chatbot

A PyTorch-based chatbot using a transformer decoder model. This project trains a sequence-to-sequence model to generate responses given input text.

---

## Features

- Transformer-based decoder for sequence generation
- Padding and masking for variable-length sequences
- Training with teacher forcing
- GPU acceleration support (CUDA)
- Optional mixed precision (FP16) for faster training
- Batch training with PyTorch DataLoader

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA (optional for GPU acceleration)
- torchvision, tqdm, numpy, pillow (PIL)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/transformer-chatbot.git
cd transformer-chatbot
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Prepare your dataset of input/output sequences (tokenized and padded).  
2. Update `vocab` and padding token index (`<PAD>`).  
3. Train the model:

```python
from train import train
train(dec, device, loader, optimizer, vocab["<PAD>"], criterion, epochs=50)
```

4. Test the model with a function like:

```python
def generate_response(model, input_seq, pad_idx, max_len=50):
    # Convert input_seq to tensor, add batch dimension, move to device
    # Run decoder and return generated tokens
    ...
```

---

## Optimization Tips

- Use GPU (`device='cuda'`) for faster training
- Use mixed precision (`torch.cuda.amp`) to reduce memory usage
- Reduce batch size or sequence length if memory is limited
- Use `num_workers>0` in DataLoader for faster data loading

---

## File Structure

```
transformer-chatbot/
│
├─ train.py             # Training loop
├─ model.py             # Transformer decoder model
├─ dataset.py           # Data preprocessing & DataLoader
├─ utils.py             # Helper functions
├─ requirements.txt
└─ README.md
```

---

## License

MIT License
