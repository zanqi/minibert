# CS 224N Default Final Project - Multitask BERT

This is the starting code for the default final project for the Stanford CS 224N class. You can find the handout [here](https://web.stanford.edu/class/cs224n/project/default-final-project-bert-handout.pdf)

In this project, you will implement some important components of the BERT model to better understanding its architecture. 
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, external libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).

## Handout

Please refer to the handout for a through description of the project and its parts.

### Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

## Write up!

### Part 1
The results from the pdf are copied here: \
Pretraining for SST: Dev Accuracy: 0.390 (0.007) \
Pretraining for CFIMDB: Dev Accuracy: 0.780 (0.002) \
Finetuning for SST: Dev Accuracy: 0.515 (0.004) \
Finetuning for CFIMDB: Dev Accuracy: 0.966 (0.007) \

Actual results: \
```
python3 classifier.py --option pretrain --device=mps --lr=1e-3
```
Pretraining for SST: Dev Accuracy: 0.394 \
Pretraining for CFIMDB: Dev Accuracy: 0.796 \
```
python3 classifier.py --option finetune --device=mps --lr=1e-5
```
Finetuning for SST: Dev Accuracy: 0.519 \
Finetuning for CFIMDB: Dev Accuracy: 0.967 \

Pretraining can accept a higher learning rate because has fewer weights to train. Less chance to diverge. Finetuning requires a lower learning rate because it is a more complex task with more weights to train. More chance to diverge. \

### Part 2