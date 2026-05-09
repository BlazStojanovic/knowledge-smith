---
arxiv: '1502.03167'
authors:
- Sergey Ioffe
- Christian Szegedy
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1502.03167.md
raw_pdf: raw/papers/1502.03167.pdf
read: false
slug: batch-normalization-accelerating-deep-network-training-by
tags:
- optimization
- deep-learning
- generalization
title: 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal
  Covariate Shift'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1502.03167v3
venue: null
year: 2015
---

# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

> *Sergey Ioffe, Christian Szegedy* — arXiv 1502.03167, 2015

## Abstract

Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1502.03167]]
- PDF (gitignored): `raw/papers/1502.03167.pdf`
- arXiv: <http://arxiv.org/abs/1502.03167v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2015-ioffe-batchnorm.md` before that tree was retired.*

Batch normalization; the stability trick that made deep vision nets trainable.
