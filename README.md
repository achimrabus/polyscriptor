# Slavistik GitLab Project

## TODO

1. Finetuning/pre-training des Transformer-ATR-Models TrOCR für kyrillische Schrift in unterschiedlichen Sprachen.
2. Installation einer eScriptorium-Instanz auf dem Server des Lab.

# Problems appeared
- **AttributeError: module 'torch' has no attribute 'get_default_device'**
- Solution: update torch and torchvision
- `pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118`

- **Transformers error**
- `ViModel.forward() got an unexpected keyword argument 'num_items_in_batch'`
- Solution: Following the forum for further steps:
https://github.com/huggingface/transformers/issues/36074

# TODOs
1. Pre-Processed the russian handwritten dataset (Splitted the dataset in two halfs);
2. Updated the augmentation (More complicated: transforms.RandomRotation and transforms.RandomAffine);
3. Updated the model cyrillic_seq2seq_trocr21 using 2 halfs of the russian dataset 1365312;
4. NOW: Training of the "kazars24/trocr-base-handwritten-ru" using the same dataset and check if the eval_loss is better.

- Training von "microsoft/trocr-base-handwritten" mit epochs=10 und batch size=2 auf den Datasets 1365312 und 6470048 ergeben Eval loss:  0.5753200054168701.