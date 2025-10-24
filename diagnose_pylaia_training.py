"""
Diagnose PyLaia training issues.
"""

import torch
from pathlib import Path
from train_pylaia import PyLaiaDataset, CRNN, collate_fn
from torch.utils.data import DataLoader
import json

def check_vocabulary():
    """Check vocabulary encoding."""
    print("="*60)
    print("VOCABULARY CHECK")
    print("="*60)
    
    dataset = PyLaiaDataset('data/pylaia_efendiev_train', augment=False)
    
    print(f"\nVocabulary size: {len(dataset.symbols)}")
    print(f"First 20 symbols: {dataset.symbols[:20]}")
    
    # Check <SPACE> token
    if '<SPACE>' in dataset.symbols:
        space_idx = dataset.char2idx['<SPACE>']
        print(f"\n<SPACE> token found at index {space_idx}")
        print(f"Maps back to: '{dataset.idx2char[space_idx]}'")
    else:
        print("\nWARNING: <SPACE> token not found in vocabulary!")
    
    # Check a sample
    img, target, text, sample_id = dataset[0]
    print(f"\nSample text: '{text}'")
    print(f"Target indices: {target.tolist()}")
    
    # Try to decode
    decoded_chars = [dataset.idx2char.get(idx.item(), '?') for idx in target]
    decoded = ''.join(decoded_chars)
    print(f"Decoded back: '{decoded}'")
    
    if text != decoded:
        print(f"\n❌ ENCODING/DECODING MISMATCH!")
        print(f"Original:  '{text}'")
        print(f"Decoded:   '{decoded}'")
        
        # Character-by-character comparison
        print("\nCharacter comparison:")
        max_len = max(len(text), len(decoded))
        for i in range(max_len):
            orig_char = text[i] if i < len(text) else '∅'
            dec_char = decoded[i] if i < len(decoded) else '∅'
            match = '✓' if orig_char == dec_char else '✗'
            print(f"  {i:3d}: '{orig_char}' vs '{dec_char}' {match}")
    else:
        print(f"\n✓ Encoding/decoding works correctly for sample 0")


def check_model_output():
    """Check model output dimensions."""
    print("\n" + "="*60)
    print("MODEL OUTPUT CHECK")
    print("="*60)
    
    dataset = PyLaiaDataset('data/pylaia_efendiev_train', augment=False)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    num_classes = len(dataset.symbols) + 1
    model = CRNN(
        img_height=128,
        num_classes=num_classes,
        cnn_filters=[12, 24, 48, 48],
        cnn_poolsize=[2, 2, 0, 2],
        rnn_hidden=256,
        rnn_layers=3,
        dropout=0.5
    )
    
    model.eval()
    
    images, targets, input_lengths, target_lengths, texts, sample_ids = next(iter(loader))
    
    print(f"\nBatch info:")
    print(f"  Images shape: {images.shape}")
    print(f"  Input widths: {input_lengths.tolist()}")
    print(f"  Target lengths: {target_lengths.tolist()}")
    print(f"  Texts: {texts}")
    
    with torch.no_grad():
        log_probs = model(images)
    
    print(f"\nModel output:")
    print(f"  Log probs shape: {log_probs.shape}")  # [width, batch, num_classes]
    print(f"  Expected: [width/8, batch={len(images)}, classes={num_classes}]")
    
    # Calculate actual sequence lengths after CNN
    actual_lengths = input_lengths // 8
    print(f"\nSequence lengths after CNN: {actual_lengths.tolist()}")
    
    # Check if lengths are valid for CTC
    for i, (seq_len, target_len) in enumerate(zip(actual_lengths, target_lengths)):
        if seq_len < target_len:
            print(f"\n❌ Sample {i}: sequence length {seq_len} < target length {target_len}")
            print(f"   This will cause CTC loss to fail!")
            print(f"   Text: '{texts[i]}'")
        else:
            print(f"✓ Sample {i}: seq_len={seq_len}, target_len={target_len}")
    
    # Try to decode
    from train_pylaia import PyLaiaTrainer
    
    class DummyTrainer:
        def __init__(self, idx2char):
            self.idx2char = idx2char
        
        def decode_predictions(self, log_probs):
            predictions = []
            _, preds = log_probs.max(2)
            preds = preds.transpose(1, 0).contiguous()
            
            for pred in preds:
                chars = []
                prev_char = None
                for idx in pred.tolist():
                    if idx == 0:
                        prev_char = None
                        continue
                    if idx == prev_char:
                        continue
                    chars.append(self.idx2char.get(idx, '?'))
                    prev_char = idx
                text = ''.join(chars)
                predictions.append(text)
            return predictions
    
    trainer = DummyTrainer(dataset.idx2char)
    preds = trainer.decode_predictions(log_probs)
    
    print(f"\nPredictions (untrained model):")
    for i, (pred, ref) in enumerate(zip(preds, texts)):
        print(f"  {i}: Pred: '{pred[:50]}...' | Ref: '{ref[:50]}...'")


def check_training_data():
    """Check training data statistics."""
    print("\n" + "="*60)
    print("TRAINING DATA CHECK")
    print("="*60)
    
    # Load both datasets
    train_dataset = PyLaiaDataset('data/pylaia_efendiev_train', augment=False)
    val_dataset = PyLaiaDataset('data/pylaia_efendiev_val', augment=False)
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # Check text lengths
    train_lengths = []
    for i in range(min(100, len(train_dataset))):
        _, target, text, _ = train_dataset[i]
        train_lengths.append(len(text))
    
    print(f"\nText length statistics (first 100 samples):")
    print(f"  Min: {min(train_lengths)}")
    print(f"  Max: {max(train_lengths)}")
    print(f"  Avg: {sum(train_lengths)/len(train_lengths):.1f}")
    
    # Check for empty texts
    empty_count = sum(1 for l in train_lengths if l == 0)
    if empty_count > 0:
        print(f"\n❌ Found {empty_count} empty texts!")
    
    # Sample some texts
    print(f"\nSample texts:")
    for i in range(5):
        _, _, text, sample_id = train_dataset[i]
        print(f"  {sample_id}: '{text[:100]}...'")


def main():
    check_vocabulary()
    check_model_output()
    check_training_data()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nIf you see ❌ errors above, that's likely the cause of high CER.")


if __name__ == '__main__':
    main()