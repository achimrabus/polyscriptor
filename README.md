# Slavistik GitLab Project

## TODO

1. Finetuning/pre-training des Transformer-ATR-Models TrOCR für kyrillische Schrift in unterschiedlichen Sprachen.
2. Installation einer eScriptorium-Instanz auf dem Server des Lab.

# Problems appeared
AttributeError: module 'torch' has no attribute 'get_default_device'
Solution: update torch and torchvision
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118

Transformers error
`ViModel.forward() got an unexpected keyword argument 'num_items_in_batch'`
Solution: Following the forum for further steps:
https://github.com/huggingface/transformers/issues/36074

# In Process
1. Pre-Processing of the russian handwritten dataset;
- Splitted the dataset in two halfs, batch size = 2.
2. Updating the augmentation;
- More complicated: transforms.RandomRotation and transforms.RandomAffine.
3. Update of the fine tuned cyrillic_seq2seq_trocr21 model (using a russian dataset 1365312).