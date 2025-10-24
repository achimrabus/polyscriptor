"""
Extended diagnostics - check actual training behavior.
"""

import torch
import torch.nn as nn
from pathlib import Path
from train_pylaia import PyLaiaDataset, CRNN, collate_fn
from torch.utils.data import DataLoader
from jiwer import cer as compute_cer

def test_single_batch_overfit():
    """Test if model can overfit on a single batch (sanity check)."""
    print("="*60)
    print("SINGLE BATCH OVERFITTING TEST")
    print("="*60)
    
    dataset = PyLaiaDataset('data/pylaia_efendiev_train', augment=False)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
    
    # Get one batch
    images, targets, input_lengths, target_lengths, texts, sample_ids = next(iter(loader))
    
    print(f"\nTest batch:")
    for i, text in enumerate(texts):
        print(f"  {i}: '{text}'")
    
    # Create model
    num_classes = len(dataset.symbols) + 1
    model = CRNN(
        img_height=128,
        num_classes=num_classes,
        cnn_filters=[12, 24, 48, 48],
        cnn_poolsize=[2, 2, 0, 2],
        rnn_hidden=256,
        rnn_layers=3,
        dropout=0.0  # Disable dropout for overfitting test
    )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Try to overfit on this single batch
    print("\nTraining on single batch (should reach near 0 loss):")
    model.train()
    
    for epoch in range(50):
        optimizer.zero_grad()
        
        log_probs = model(images)
        
        # Calculate input lengths (after 3 pooling layers)
        seq_lengths = input_lengths // 8
        seq_lengths = torch.clamp(seq_lengths, min=1)
        
        loss = criterion(log_probs, targets, seq_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            # Decode predictions
            model.eval()
            with torch.no_grad():
                log_probs = model(images)
                _, preds = log_probs.max(2)
                preds = preds.transpose(1, 0).contiguous()
                
                predictions = []
                for pred in preds:
                    chars = []
                    prev_char = None
                    for idx in pred.tolist():
                        if idx == 0:
                            prev_char = None
                            continue
                        if idx == prev_char:
                            continue
                        chars.append(dataset.idx2char.get(idx, '?'))
                        prev_char = idx
                    predictions.append(''.join(chars))
                
                # Calculate CER
                cer = compute_cer(list(texts), predictions)
                
                print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, CER={cer*100:.1f}%")
                
                if epoch + 1 == 50:
                    print("\nFinal predictions:")
                    for i, (pred, ref) in enumerate(zip(predictions, texts)):
                        print(f"  {i}: Pred: '{pred}'")
                        print(f"      Ref:  '{ref}'")
            
            model.train()
    
    if loss.item() > 0.1 or cer > 0.05:
        print("\n❌ PROBLEM: Model cannot overfit single batch!")
        print("   This suggests a fundamental issue with model architecture or loss.")
    else:
        print("\n✓ Model can overfit - architecture seems OK")


def check_image_preprocessing():
    """Check if images are preprocessed correctly."""
    print("\n" + "="*60)
    print("IMAGE PREPROCESSING CHECK")
    print("="*60)
    
    dataset = PyLaiaDataset('data/pylaia_efendiev_train', augment=False)
    
    # Check a few samples
    for i in range(3):
        img, target, text, sample_id = dataset[i]
        
        print(f"\nSample {i} ({sample_id}):")
        print(f"  Text: '{text}'")
        print(f"  Image shape: {img.shape}")  # Should be [1, 128, width]
        print(f"  Image dtype: {img.dtype}")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Check if normalized properly
        if img.min() < -2 or img.max() > 2:
            print(f"  ⚠️  Image values outside expected range [-2, 2]")
        
        # Check dimensions
        if img.shape[1] != 128:
            print(f"  ❌ Height is {img.shape[1]}, expected 128!")


def check_ctc_loss_inputs():
    """Check CTC loss input validity."""
    print("\n" + "="*60)
    print("CTC LOSS INPUT CHECK")
    print("="*60)
    
    dataset = PyLaiaDataset('data/pylaia_efendiev_train', augment=False)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    
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
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Check several batches
    problems = 0
    total_batches = 0
    
    for batch_idx, (images, targets, input_lengths, target_lengths, texts, sample_ids) in enumerate(loader):
        if batch_idx >= 10:  # Check first 10 batches
            break
        
        total_batches += 1
        
        with torch.no_grad():
            log_probs = model(images)
        
        seq_lengths = input_lengths // 8
        seq_lengths = torch.clamp(seq_lengths, min=1)
        
        # Check each sample in batch
        for i in range(len(images)):
            seq_len = seq_lengths[i].item()
            target_len = target_lengths[i].item()
            
            if seq_len < target_len:
                print(f"\n❌ Batch {batch_idx}, Sample {i} ({sample_ids[i]}):")
                print(f"   Input width: {input_lengths[i]}, after CNN: {seq_len}")
                print(f"   Target length: {target_len}")
                print(f"   Text ({len(texts[i])} chars): '{texts[i]}'")
                print(f"   Ratio: {seq_len/target_len:.2f}")
                problems += 1
        
        # Try loss calculation
        try:
            loss = criterion(log_probs, targets, seq_lengths, target_lengths)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ Batch {batch_idx}: Loss is NaN or Inf!")
                problems += 1
        except Exception as e:
            print(f"\n❌ Batch {batch_idx}: CTC loss failed with error: {e}")
            problems += 1
    
    print(f"\nChecked {total_batches} batches, found {problems} problems")
    
    if problems > 0:
        print("\n⚠️  Found CTC input issues - this could cause training problems!")


def check_learning_rate_and_optimizer():
    """Check if learning rate is appropriate."""
    print("\n" + "="*60)
    print("LEARNING RATE CHECK")
    print("="*60)
    
    print("\nCurrent settings:")
    print("  Learning rate: 0.0003")
    print("  Optimizer: RMSprop")
    print("  Weight decay: 0.0")
    
    print("\nRecommendations:")
    print("  • 0.0003 is reasonable for RMSprop")
    print("  • Consider trying AdamW with LR=0.001")
    print("  • Add gradient clipping (max_norm=5.0) ✓ Already implemented")


def main():
    check_image_preprocessing()
    check_ctc_loss_inputs()
    test_single_batch_overfit()
    check_learning_rate_and_optimizer()
    
    print("\n" + "="*60)
    print("EXTENDED DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()