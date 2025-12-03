---
authors:
- affiliation: "\xC9cole Pratique des Hautes \xC9tudes, PSL University"
  name: Benjamin Kiessling
  orcid: https://orcid.org/0000-0001-9543-7827
citation: https://doi.org/10.1109/ICFHR2020.2020.00064
datasets:
- https://doi.org/10.5281/zenodo.3568023a
id: 10.5281/zenodo.14602569
language:
- und
license: Apache-2.0
model_type:
- segmentation
script:
- Zyyy
software_hints:
- kind=vgsl
software_name: kraken
summary: General segmentation model for print and handwriting
tags:
- multiscriptal
---
# BLLA base model

This is the default segmentation model shipped with the kraken ATR engine.

## Uses

The model performs reasonably well on most non-fragmentary handwritten and
machine-printed document pages of moderate complexity. If out of the box
performance is unsatisfactory or more fine-grained segmentation taxonomies are
desired it can also serve as a good foundation model for fine-tuning.

## Normalization and Transformations 

Region and line types in the training dataset have been merged into single classes.

## Biases and Limitations

The training corpus skews heavily towards Latin script handwritten documents
with relatively simple layouts. Therefore, the model's accuracy on scripts
dissimilar to Latin is fairly low. Some features of pages where segmentation
accuracy is frequently unsatisfactory:

- Tilted and vertical writing
- Results on writing embedded in ecoration
- Closely typeset lines such as those found in newspaper columns resulting in merging

An idiosyncracy of the model is that no matter the position of the
calligraphic/typographic baseline (top-, center-, or baseline) a baseline will
be detected. Hebrew script lines will there be detected with base- not
toplines. This is generally not an issue for recognition models that have been
trained with the same line position.

## How to Get Started with the Model

Install the `kraken` ATR package following the instruction
[here](https://kraken.re). A quick start explaining the use of the segmenter
can also be found there.

## Training

The model has been trained for 50 epochs with kraken's default hyperparameters
on the [cBAD 2019](https://zenodo.org/records/3568023) dataset including the
validation and test set.


## Citation

**BibTeX:**

@INPROCEEDINGS{9257770,
  author={Kiessling, Benjamin},
  booktitle={2020 17th International Conference on Frontiers in Handwriting Recognition (ICFHR)}, 
  title={A Modular Region and Text Line Layout Analysis System}, 
  year={2020},
  volume={},
  number={},
  pages={313-318},
  keywords={Optical character recognition software;Layout;Semantics;Task analysis;Neural networks;Feature extraction;Particle separators;layout analysis;region detection;historical document analysis;artificial neural networks},
  doi={10.1109/ICFHR2020.2020.00064}}

