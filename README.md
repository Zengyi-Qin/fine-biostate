# Learning Fine-grained Estimate of Biological States from Coarse Labels

This repository contains an official implementation of **Learning Fine-grained Estimate of Biological States from Coarse Labels**

### Install
Create an virtual environment with Python 3.6 using Anaconda:
```bash
conda create - n bioenet python=3.6
```

Download and extract the simulated sEMG dataset:
```bash
wget https://cloud.tsinghua.edu.cn/f/fd012a23ef894c949e2d/?dl=1 -O data.zip
unzip data.zip
```

Install the requirements:
```bash
pip install - r requirements.txt
```

### Training and Testing
Run the following command for training. The testing will be performed during training.
```bash
python train_test.py
```
