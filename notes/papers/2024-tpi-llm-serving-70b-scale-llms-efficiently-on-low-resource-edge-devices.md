---
arxiv: '2410.00531'
authors:
- Zonghang Li
- Wenjiao Feng
- Mohsen Guizani
- Hongfang Yu
created: '2026-06-01'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2410.00531
  raw: '[[raw/papers/md/2024-tpi-llm-serving-70b-scale-llms-efficiently-on-low-resource-edge-devices]]'
  source: https://arxiv.org/abs/2410.00531
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-tpi-llm-serving-70b-scale-llms-efficiently-on-low-resource-edge-devices.md
raw_pdf: raw/papers/pdf/2024-tpi-llm-serving-70b-scale-llms-efficiently-on-low-resource-edge-devices.pdf
read: false
slug: tpi-llm-serving-70b-scale-llms-efficiently-on-low-resource-edge-devices
tags:
- type/paper
- status/stub
title: 'TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices'
type: note
updated: '2026-06-01'
year: 2024
---

# TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices

> *Zonghang Li, Wenjiao Feng, Mohsen Guizani, Hongfang Yu* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

Large model inference is shifting from cloud to edge due to concerns about the privacy of user interaction data. However, edge devices often struggle with limited computing power, memory, and bandwidth, requiring collaboration across multiple devices to run and speed up LLM inference. Pipeline parallelism, the mainstream solution, is inefficient for single-user scenarios, while tensor parallelism struggles with frequent communications. In this paper, we argue that tensor parallelism can be more effective than pipeline on low-resource devices, and present a compute- and memory-efficient tensor parallel inference system, named TPI-LLM, to serve 70B-scale models. TPI-LLM keeps sensitive raw data local in the users' devices and introduces a sliding window memory scheduler to dynamically manage layer weights during inference, with disk I/O latency overlapped with the computation and communication. This allows larger models to run smoothly on memory-limited devices. We analyze the communication bottleneck and find that link latency, not bandwidth, emerges as the main issue, so a star-based allreduce algorithm is implemented. Through extensive experiments on both emulated and real testbeds, TPI-LLM demonstrated over 80% less time-to-first-token and token latency compared to Accelerate, and over 90% compared to Transformers and Galaxy, while cutting the peak memory footprint of Llama 2-70B by 90%, requiring only 3.1 GB of memory for 70B-scale models.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2410.00531>
- PDF: [[raw/papers/pdf/2024-tpi-llm-serving-70b-scale-llms-efficiently-on-low-resource-edge-devices.pdf]]
- Raw markdown: [[raw/papers/md/2024-tpi-llm-serving-70b-scale-llms-efficiently-on-low-resource-edge-devices]]
