#!/usr/bin/env bash

# Have to run in bash

conda create -n cs224n_dfp python=3.8
conda activate cs224n_dfp

conda install pytorch torchvision torchaudio -c pytorch
pip install tqdm
pip install requests==2.25.1
pip install importlib-metadata
pip install filelock==3.0.12
pip install scikit-learn
pip install tokenizers==0.10.1
pip install explainaboard_client==0.0.7
