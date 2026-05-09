---
arxiv: '1706.06978'
authors:
- Guorui Zhou
- Chengru Song
- Xiaoqiang Zhu
- Ying Fan
- Han Zhu
- Xiao Ma
- Yanghui Yan
- Junqi Jin
- Han Li
- Kun Gai
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1706.06978.md
raw_pdf: raw/papers/1706.06978.pdf
read: false
slug: deep-interest-network-for-click-through-rate-prediction
tags: []
title: Deep Interest Network for Click-Through Rate Prediction
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1706.06978v4
venue: null
year: 2017
---

# Deep Interest Network for Click-Through Rate Prediction

> *Guorui Zhou, Chengru Song, Xiaoqiang Zhu…* — arXiv 1706.06978, 2017

## Abstract

Click-through rate prediction is an essential task in industrial applications, such as online advertising. Recently deep learning based models have been proposed, which follow a similar Embedding\&MLP paradigm. In these methods large scale sparse input features are first mapped into low dimensional embedding vectors, and then transformed into fixed-length vectors in a group-wise manner, finally concatenated together to fed into a multilayer perceptron (MLP) to learn the nonlinear relations among features. In this way, user features are compressed into a fixed-length representation vector, in regardless of what candidate ads are. The use of fixed-length vector will be a bottleneck, which brings difficulty for Embedding\&MLP methods to capture user's diverse interests effectively from rich historical behaviors. In this paper, we propose a novel model: Deep Interest Network (DIN) which tackles this challenge by designing a local activation unit to adaptively learn the representation of user interests from historical behaviors with respect to a certain ad. This representation vector varies over different ads, improving the expressive ability of model greatly. Besides, we develop two techniques: mini-batch aware regularization and data adaptive activation function which can help training industrial deep networks with hundreds of millions of parameters. Experiments on two public datasets as well as an Alibaba real production dataset with over 2 billion samples demonstrate the effectiveness of proposed approaches, which achieve superior performance compared with state-of-the-art methods. DIN now has been successfully deployed in the online display advertising system in Alibaba, serving the main traffic.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1706.06978]]
- PDF: `raw/papers/1706.06978.pdf`
- arXiv: <http://arxiv.org/abs/1706.06978v4>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2018-zhou-din]]
