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

# Done
1. Pre-Processed the russian handwritten dataset (Splitted the dataset in two halfs);
2. Updated the augmentation (More complicated: transforms.RandomRotation and transforms.RandomAffine);
3. Updated the model cyrillic_seq2seq_trocr21 using 2 halfs of the russian dataset 1365312;

# TODO
1. Test another script: https://github.com/wjbmattingly/trocr-train .
2. Test finetuning using QWEN https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl .

# Resultate
Training of the "kazars24/trocr-base-handwritten-ru" using all 3 datasets. <br />
CER = 0.253278 <br />
Shows well results on all three datasets: Inference is in the Fine_Tune_TrOCR_transfVersion.ipynb <br />
The latest Model: https://www.dropbox.com/scl/fi/umth7s1l619ok693l9xcy/seq2seq_mixed.zip?rlkey=yibenn1kewdyxunuyxzfr7ua0&st=a3ubnjca&dl=0