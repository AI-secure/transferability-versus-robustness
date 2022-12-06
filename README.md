# Adversarially Robust Models may not Transfer Better

This repo provides the implementation of the paper [Adversarially Robust Models may not Transfer Better: Sufficient Conditions for Domain Transferability from the View of Regularization](https://arxiv.org/pdf/2202.01832.pdf).

## Prerequisites

The code successfully run on Python 3.6 + PyTorch 1.8.1. The PyTorch package needs to be installed based on different hardware environments. Other packages can be installed by `pip install -r requirements.txt`.

## Training

```python {train,train_aug,train_lastot,train_reg}.py```
+ The hyper-parameters are specified within each script.

## Evaluation

```python {eval_attack,eval_transfer}.py```
+ The hyper-parameters are specified within each script.

