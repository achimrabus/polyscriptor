# Slavistik GitLab Project

## TODO

1. Finetuning/pre-training des Transformer-ATR-Models TrOCR für kyrillische Schrift in unterschiedlichen Sprachen.
2. Installation einer eScriptorium-Instanz auf dem Server des Lab.

# Current problem
transformers error
`ViModel.forward() got an unexpected keyword argument 'num_items_in_batch'`
Following the forum for further steps:
https://github.com/huggingface/transformers/issues/36074

# In Process
1. Pre-Processing of the russian handwritten dataset;
2. Updating the augmentation;
3. Update of the fine tuned cyrillic_seq2seq_trocr21 model (on base of trocr-base-handwritten and ukrainian dataset).