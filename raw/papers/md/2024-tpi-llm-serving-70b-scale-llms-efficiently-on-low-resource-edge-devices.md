---
arxiv: '2410.00531'
authors:
- Zonghang Li
- Wenjiao Feng
- Mohsen Guizani
- Hongfang Yu
parser: ar5iv
retrieved: '2026-06-01'
source: paper
title: 'TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices'
url: https://arxiv.org/abs/2410.00531
year: 2024
---

[2410.00531] TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices



# TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices

Zonghang Li
  
Department of Machine Learning
  
MBZUAI
  
zonghang.li@mbzuai.ac.ae
  
&Wenjiao Feng
  
School of Info & Comm Engineering
  
UESTC
  
wenjiaofeng@std.uestc.edu.cn
  
\ANDMohsen Guizani
  
Department of Machine Learning
  
MBZUAI
  
mohsen.guizani@mbzuai.ac.ae
  
&&Hongfang Yu
  
School of Info & Comm Engineering
  
UESTC
  
yuhf@uestc.edu.cn
Zonghang Li and Wenjiao Feng contribute equally.

###### Abstract

Large model inference is shifting from cloud to edge due to concerns about the privacy of user interaction data. However, edge devices often struggle with limited computing power, memory, and bandwidth, requiring collaboration across multiple devices to run and speed up LLM inference. Pipeline parallelism, the mainstream solution, is inefficient for single-user scenarios, while tensor parallelism struggles with frequent communications. In this paper, we argue that tensor parallelism can be more effective than pipeline on low-resource devices, and present a compute- and memory-efficient tensor parallel inference system, named TPI-LLM, to serve 70B-scale models. TPI-LLM keeps sensitive raw data local in the users’ devices and introduces a sliding window memory scheduler to dynamically manage layer weights during inference, with disk I/O latency overlapped with the computation and communication. This allows larger models to run smoothly on memory-limited devices. We analyze the communication bottleneck and find that link latency, not bandwidth, emerges as the main issue, so a star-based allreduce algorithm is implemented. Through extensive experiments on both emulated and real testbeds, TPI-LLM demonstrated over 80% less time-to-first-token and token latency compared to Accelerate, and over 90% compared to Transformers and Galaxy, while cutting the peak memory footprint of Llama 2-70B by 90%, requiring only 3.1 GB of memory for 70B-scale models.

## 1 Introduction

Recently, Large Language Models (LLMs) have been widely deployed in the cloud for inference. User inputs are uploaded to the cloud, where high-performance GPUs are used to compute output sequences, and then sent back to user devices for display. This process poses privacy risks, as user prompts are exposed to network intermediaries and clouds.
Therefore, there is an increasing need to shift LLM services to the network edge, such as on laptops, hand phones, tablets, and desktop computers. However, edge devices have very limited memory (4-16 GB) and computing power (often CPU-only). Even with quantization, running a Llama 3.1-70B model still requires at least 40 GB of memory, which far exceeds the capacity of most edge devices. Besides, running Bert-L on one Nano-M device results in a latency that is 120×\times longer than on one A100 GPU. This gap requires the use of more edge devices to support and speed up LLM inference on the network edge.

While advanced LLM serving systems (Shoeybi et al., [2019](#bib.bib18); Rasley et al., [2020](#bib.bib17); Li et al., [2023](#bib.bib12); Agrawal et al., [2024](#bib.bib1); Miao et al., [2024](#bib.bib16)) have been designed for high-performance GPU clusters, recent efforts (Zhang et al., [2024](#bib.bib23); Mei et al., [2024](#bib.bib15); Borzunov et al., [2024](#bib.bib3)) are adapting these systems to edge environments, by adaptively partitioning model between edge devices and optimizing schedulers to boost token throughput. However, in smart home scenarios like smart speaker, edge LLM systems often handle one user request at a time, making them degrade from pipeline to model parallelism and leaving devices idle most of the time. Thus, tensor parallelism is preferred for better efficiency. For instance, Ye et al. ([2024](#bib.bib22)) combine tensor and sequence parallelism to reduce token latency and Wei et al. ([2024](#bib.bib20)) introduce block parallelism to restructure Transformer layers.

However, even with 8 devices sharing the load, running full-precision Llama 2-70B still requires 35 GB per device, memory remains a shortage. Solutions like memory block paging (Kwon et al., [2023](#bib.bib9)) and optimized KVCache storage (Jin et al., [2023](#bib.bib8); Lee et al., [2024](#bib.bib10)) help schedule data between GPUs and CPUs, but unfortunately, GPUs are not available on most edge devices. As a popular alternative, Accelerate (Gugger et al., [2022](#bib.bib7)) can offload model data from a CPU to a disk to run larger models, but its blocking I/O drastically slows inference, with token latency increases to 30 seconds per token on Llama 3.1-8B.

In this work, we analyze why tensor parallelism is more effective than model parallelism on low-resource edge devices and present TPI-LLM, a computing- and memory-efficient tensor parallel inference framework to serve LLM models. Constrained by the high link latency, a star-based allreduce algorithm is implemented. To address the memory shortage, a sliding window memory scheduler is further introduced. We build a prototype of TPI-LLM with 3K LoC and two testbeds using Klonet (Ma et al., [2024](#bib.bib14)) and 4 laptops. Extensive results on Llama 3.1-8B/70B (Dubey et al., [2024](#bib.bib5)), Llama 2-3B/7B/13B/70B (Touvron et al., [2023](#bib.bib19)) and Yi-34B (AI et al., [2024](#bib.bib2)) demonstrate the significant reduction of the memory footprint and faster inference speed compared to Transformers (Wolf et al., [2020](#bib.bib21)), Acclerate (Gugger et al., [2022](#bib.bib7)), and Galaxy (Ye et al., [2024](#bib.bib22)).

We summarize the main contributions of this work as follows:

* •

  We design a TPI-LLM for edge LLM serving, which keeps prompt privacy in mind to allow edge devices with limited computing power collaborate to deliver faster inference.
* •

  We find that network bandwidth is no longer an issue. Instead, link latency causes high delays in advanced allreduce algorithms. Thus, a star-based allreduce algorithm is implemented, which greatly outperforms ring- and tree-based methods.
* •

  We introduce a sliding window memory scheduler, which asynchronously loads and unloads layer weights and overlaps disk I/O latency with computations and communications, enabling the inference of larger models on low-memory devices.
* •

  We prototype TPI-LLM and show that it reduces time-to-first-token and token latency by over 80% compared to Accelerate and over 90% compared to Transformers and Galaxy. It serves Llama 2-70B with a peak memory footprint of 3.1 GB across 8 low-resource devices.

## 2 Observations and Motivations

Before presenting our TPI-LLM system, we address two questions that guide our design:

Q1: On low-resource edge devices, which dominate inference time: computation or communication? Which is more efficient, tensor parallelism or model parallelism?

On the network edge, the balance between computation and communication differs from that in high-performance GPU clusters. To determine whether tensor or model parallelism offers more benefits, it is essential to identify which—computation or communication—takes up more time. For this purpose, we examine the Llama 3.1-8B model on a LAN network with 4 laptops of 8 cores. The network bandwidth between them is 178 Mbps, and the devices implement allreduce communications using a parameter server architecture (Li et al., [2014](#bib.bib11)).

(a)

(b)

(c)

Figure 1: Comparison of (a,b) tensor and model parallelism in terms of computational and communication time and (c) memory footprint each device with increasing tensor parallel nodes.

Figures [1(a)](#S2.F1.sf1 "In Figure 1 ‣ 2 Observations and Motivations ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") and [1(b)](#S2.F1.sf2 "In Figure 1 ‣ 2 Observations and Motivations ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") show the timeline and computing-communication time ratio for model and tensor parallelism during inference. In model parallelism, communication accounts for only 2% of the time, with most spent on computation. However, when one device is computing, others are idle, creating pipeline bubbles and resource waste. In tensor parallelism, communication rises to 70%, but all devices compute simultaneously, and the speed boost outweighs the communication cost, leading to less overall inference time. This makes tensor parallelism the preferred choice.

Q2: Is tensor parallelism enough for edge LLM serving?

Tensor parallelism does reduce memory footprint each device by sharing model parameters across multiple devices, but it doesn’t fully address the memory shortage. Figure [1(c)](#S2.F1.sf3 "In Figure 1 ‣ 2 Observations and Motivations ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") shows that even with 4 tensor parallel nodes, memory footprint remains at 12 GB—still too high for most edge devices. This is because memory footprint includes not just model parameters but also intermediate results, key value cache, libraries, etc., causing the actual usage to exceed the theoretical value. Besides, other apps on the device also compete for memory, which worsens the shortage. Thus, even with tensor parallelism, a memory scheduler is still needed to avoid out-of-memory (OOM) issues.

## 3 TPI-LLM Framework with Sliding Window Memory Scheduling

In a typical inference workflow, many users send their prompts to a cloud-based service. These prompts are pooled and scheduled in batches, undergoing dozens of Transformer layers, and converted into probabilities to predict the next token. This process repeats until the generated sequence is finished. While the fundamental workflow on the cloud and edge are similar, key differences arise:

(a) Keep prompts and generated sequences on users’ device.
In a cloud setup, user prompts are sent to remote servers for processing, which result in exposure of private data. Edge LLM serving systems are required to keep prompts and generated sequences in users’ own devices to ensure raw data never get exposed to external unknown environments.

(b) More single-prompt serving. Current LLM serving systems are typically optimized for batched prompts using pipeline scheduling. However, these optimizations lead to resource underutilization in edge scenarios like smart speakers, where only one prompt is processed at a time.

(c) Low-resource devices without CUDA support.
Edge devices, unlike cloud GPUs, have very limited memory and low computing power. Many of them lack CUDA support or do not have GPUs at all, and they often prioritize full precision to ensure faster computations.

### 3.1 The Parallel Framework Design of TPI-LLM System

Figure 2: Overview of the TPI-LLM parallel framework.

The proposed tensor parallel inference system (TPI-LLM) tackles these challenges by using a tensor parallel framework that distributes attention heads across multiple nodes. As depicted in Figure [2](#S3.F2 "Figure 2 ‣ 3.1 The Parallel Framework Design of TPI-LLM System ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), it involves a master node, typically the user’s device that initiates the prompt, and several worker nodes that share the computational load. Their pseudo codes are given in Algorithms [1](#algorithm1 "In 3.1 The Parallel Framework Design of TPI-LLM System ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") and [2](#algorithm2 "In 3.1 The Parallel Framework Design of TPI-LLM System ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices").

Step 1: The master node partitions and distributes model weights.
Before inference begins, the master node partitions the pretrained model weights 𝑾𝑾{\bm{W}}, such as attention heads and FFN weights, among the worker nodes. Workers with greater computing power and larger memory are allocated more attention heads and FFN weights. This ensures no single device bears the full burden.

Step 2: The master node initiates prompt and broadcast the input embedding to workers.
The inference process starts at the master node, where a user prompt is tokenized into a list of token indices 𝒙𝒙{\bm{x}} and then transformed into input embeddings 𝑯0=𝒙​𝑾𝚎𝚖𝚋superscript𝑯0𝒙subscript𝑾𝚎𝚖𝚋{\bm{H}}^{0}={\bm{x}}{\bm{W}}\_{\mathtt{emb}}. The embedding is then broadcast to all worker nodes 𝑯𝚏𝚏𝚗0=𝑯0subscriptsuperscript𝑯0𝚏𝚏𝚗superscript𝑯0{\bm{H}}^{0}\_{\mathtt{ffn}}={\bm{H}}^{0} to initiate the tensor parallel workflow.

Step 3: All nodes perform tensor parallel computing.
The tensor parallel computing follows a cycle of four operations: attention computing, allreduce, FFN computing, and allreduce. These operations together constitute a Transformer block. Devices compute attention and FFN with partitioned weights in parallel, reducing the computing delays on low-power devices.

In the attention computation phase of the l𝑙l-th Transformer block, device hℎh processes only a subset of attention heads 𝑸h,l=𝑯𝚗𝚘𝚛𝚖l​𝑾Qh,l,𝑲h,l=𝑯𝚗𝚘𝚛𝚖l​𝑾Kh,l,𝑽h,l=𝑯𝚗𝚘𝚛𝚖l​𝑾Vh,lformulae-sequencesuperscript𝑸

ℎ𝑙subscriptsuperscript𝑯𝑙𝚗𝚘𝚛𝚖superscriptsubscript𝑾𝑄

ℎ𝑙formulae-sequencesuperscript𝑲

ℎ𝑙subscriptsuperscript𝑯𝑙𝚗𝚘𝚛𝚖superscriptsubscript𝑾𝐾

ℎ𝑙superscript𝑽

ℎ𝑙subscriptsuperscript𝑯𝑙𝚗𝚘𝚛𝚖superscriptsubscript𝑾𝑉

ℎ𝑙{\bm{Q}}^{h,l}={\bm{H}}^{l}\_{\mathtt{norm}}{\bm{W}}\_{Q}^{h,l},{\bm{K}}^{h,l}={\bm{H}}^{l}\_{\mathtt{norm}}{\bm{W}}\_{K}^{h,l},{\bm{V}}^{h,l}={\bm{H}}^{l}\_{\mathtt{norm}}{\bm{W}}\_{V}^{h,l}, where 𝑯𝚗𝚘𝚛𝚖l=𝚗𝚘𝚛𝚖​(𝑯𝚏𝚏𝚗l−1)subscriptsuperscript𝑯𝑙𝚗𝚘𝚛𝚖𝚗𝚘𝚛𝚖superscriptsubscript𝑯𝚏𝚏𝚗𝑙1{\bm{H}}^{l}\_{\mathtt{norm}}=\mathtt{norm}({\bm{H}}\_{\mathtt{ffn}}^{l-1}) is the normed hidden state and weight partitions 𝑾Qh,l,𝑾Kh,l,𝑾Vh,l

superscriptsubscript𝑾𝑄

ℎ𝑙superscriptsubscript𝑾𝐾

ℎ𝑙superscriptsubscript𝑾𝑉

ℎ𝑙{\bm{W}}\_{Q}^{h,l},{\bm{W}}\_{K}^{h,l},{\bm{W}}\_{V}^{h,l} are downloaded from the master node in Step 1. Once 𝑸h,l,𝑲h,l,𝑽h,l

superscript𝑸

ℎ𝑙superscript𝑲

ℎ𝑙superscript𝑽

ℎ𝑙{\bm{Q}}^{h,l},{\bm{K}}^{h,l},{\bm{V}}^{h,l} are computed, we apply the scaled dot-product attention to calculate the attention score, and the result is then synchronized across devices:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑯𝚊𝚝𝚝𝚗l=𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎​(𝚜𝚘𝚏𝚝𝚖𝚊𝚡​(𝑸h,l​(𝑲h,l)⊤d)​𝑽h,l)+𝑯𝚏𝚏𝚗l−1,superscriptsubscript𝑯𝚊𝚝𝚝𝚗𝑙𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎𝚜𝚘𝚏𝚝𝚖𝚊𝚡superscript𝑸  ℎ𝑙superscriptsuperscript𝑲  ℎ𝑙top𝑑superscript𝑽  ℎ𝑙superscriptsubscript𝑯𝚏𝚏𝚗𝑙1{\bm{H}}\_{\mathtt{attn}}^{l}=\mathtt{all\\_reduce}(\mathtt{softmax}(\frac{{\bm{Q}}^{h,l}({\bm{K}}^{h,l})^{\top}}{\sqrt{d}}){\bm{V}}^{h,l})+{\bm{H}}\_{\mathtt{ffn}}^{l-1}, |  | (1) |

where d𝑑d is the dimension for attention head. Here, attention is computed in parallel across devices, followed by an allreduce to aggregate their hidden states and a shortcut connection. The key-value pair (𝑲h,l,𝑽h,l)superscript𝑲

ℎ𝑙superscript𝑽

ℎ𝑙({\bm{K}}^{h,l},{\bm{V}}^{h,l}) is cached locally on device hℎh to reduce redundant computations. This distributed KVCache partitions the cache across devices, so memory cost is reduced on individual device.

After the attention computation and allreduce, the process continues with the FFN computation:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑯𝚏𝚏𝚗l=𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎​(𝑾𝚍𝚘𝚠𝚗h,l⋅(σ​(𝑾𝚐𝚊𝚝𝚎h,l⋅𝑯𝚗𝚘𝚛𝚖l)⊙(𝑾𝚞𝚙h,l⋅𝑯𝚗𝚘𝚛𝚖l)))+𝑯𝚊𝚝𝚝𝚗l,superscriptsubscript𝑯𝚏𝚏𝚗𝑙𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎⋅superscriptsubscript𝑾𝚍𝚘𝚠𝚗  ℎ𝑙direct-product𝜎⋅superscriptsubscript𝑾𝚐𝚊𝚝𝚎  ℎ𝑙superscriptsubscript𝑯𝚗𝚘𝚛𝚖𝑙⋅superscriptsubscript𝑾𝚞𝚙  ℎ𝑙superscriptsubscript𝑯𝚗𝚘𝚛𝚖𝑙superscriptsubscript𝑯𝚊𝚝𝚝𝚗𝑙{\bm{H}}\_{\mathtt{ffn}}^{l}=\mathtt{all\\_reduce}({\bm{W}}\_{\mathtt{down}}^{h,l}\cdot(\sigma({\bm{W}}\_{\mathtt{gate}}^{h,l}\cdot{\bm{H}}\_{\mathtt{norm}}^{l})\odot({\bm{W}}\_{\mathtt{up}}^{h,l}\cdot{\bm{H}}\_{\mathtt{norm}}^{l})))+{\bm{H}}\_{\mathtt{attn}}^{l}, |  | (2) |

where FFN weights 𝑾𝚐𝚊𝚝𝚎h,l,𝑾𝚞𝚙h,l,𝑾𝚍𝚘𝚠𝚗h,l

superscriptsubscript𝑾𝚐𝚊𝚝𝚎

ℎ𝑙superscriptsubscript𝑾𝚞𝚙

ℎ𝑙superscriptsubscript𝑾𝚍𝚘𝚠𝚗

ℎ𝑙{\bm{W}}\_{\mathtt{gate}}^{h,l},{\bm{W}}\_{\mathtt{up}}^{h,l},{\bm{W}}\_{\mathtt{down}}^{h,l} are also partitioned weights, 𝑯𝚗𝚘𝚛𝚖l=𝚗𝚘𝚛𝚖​(𝑯𝚊𝚝𝚝𝚗l)superscriptsubscript𝑯𝚗𝚘𝚛𝚖𝑙𝚗𝚘𝚛𝚖superscriptsubscript𝑯𝚊𝚝𝚝𝚗𝑙{\bm{H}}\_{\mathtt{norm}}^{l}=\mathtt{norm}({\bm{H}}\_{\mathtt{attn}}^{l}), σ𝜎\sigma represents the activation function such as SiLU (Elfwing et al., [2018](#bib.bib6)). Similar to the attention computation stage, the FFN is computed in parallel, followed by an allreduce and a shortcut connection.

Step 4: The master node reduces tensor parallel results and calculates the next token. After each node hℎh completes its part of computation within the backbone network, the result is sent to the master node. The summed results 𝑯𝚏𝚏𝚗Lsuperscriptsubscript𝑯𝚏𝚏𝚗𝐿{\bm{H}}\_{\mathtt{ffn}}^{L} are then passed through a task head 𝑾𝚑𝚎𝚊𝚍subscript𝑾𝚑𝚎𝚊𝚍{\bm{W}}\_{\mathtt{head}} and softmax to obtain the probability distribution of the next token 𝒛=𝚜𝚘𝚏𝚝𝚖𝚊𝚡​(𝑯𝚏𝚏𝚗L​𝑾𝚑𝚎𝚊𝚍)𝒛𝚜𝚘𝚏𝚝𝚖𝚊𝚡subscriptsuperscript𝑯𝐿𝚏𝚏𝚗subscript𝑾𝚑𝚎𝚊𝚍{\bm{z}}=\mathtt{softmax}({\bm{H}}^{L}\_{\mathtt{ffn}}{\bm{W}}\_{\mathtt{head}}), which is then sampled. Steps 2 to 4 repeat until an EOS token is generated or the length limit is reached.

TPI-LLM provides three benefits: (i) The user prompt {𝒙1,𝒙2,⋯}subscript𝒙1subscript𝒙2⋯\{{\bm{x}}\_{1},{\bm{x}}\_{2},\cdots\} and generated sequence {z1∼𝒛1,z2∼𝒛2,⋯}formulae-sequencesimilar-tosubscript𝑧1subscript𝒛1similar-tosubscript𝑧2

subscript𝒛2⋯\{z\_{1}\sim{\bm{z}}\_{1},z\_{2}\sim{\bm{z}}\_{2},\cdots\} are processed only on the master node, keeping them hidden from workers. Even if workers reverse-engineer input embeddings 𝑯0superscript𝑯0{\bm{H}}^{0}, they cannot recover the raw prompt 𝒙𝒙{\bm{x}} or next token z∼𝒛similar-to𝑧𝒛z\sim{\bm{z}} since the weights of input embedding 𝑾𝚎𝚖𝚋subscript𝑾𝚎𝚖𝚋{\bm{W}}\_{\mathtt{emb}} and task head 𝑾𝚑𝚎𝚊𝚍subscript𝑾𝚑𝚎𝚊𝚍{\bm{W}}\_{\mathtt{head}} reside solely on master. (ii) The inference speed is often limited by the computational latency, but in TPI-LLM, it is accelerated via parallel computing. (iii) Unlike other systems that use a mix of communication primitives (reduce & broadcast (Shoeybi et al., [2019](#bib.bib18)), reducescatter & allgather (Ye et al., [2024](#bib.bib22)), etc.), TPI-LLM standardizes communications to allreduce. This enhances compatibility with broader communication libraries like PS-LITE (Chen et al., [2015](#bib.bib4)) and NetStorm (Li et al., [2024](#bib.bib13)), leveraging their optimized implementations for edge conditions.

1
Split and distribute pretrained weight files to worker nodes;

2
Tokenize user prompt into indices;

3
Start memory scheduler;

4
while *generated sequence not finished* do

5      
Preprocess: Convert indices to input and position embeddings, calculate causal mask and cache position;

6      
Broadcast: Send embeddings, causal mask, and cache position to workers;

7      
foreach *decoder layer l𝑙l* do

8            
Attention: Execute layernorm, self-attention, and store (Kl0,Vl0)superscriptsubscript𝐾𝑙0superscriptsubscript𝑉𝑙0(K\_{l}^{0},V\_{l}^{0}) in KVCache 𝒟0superscript𝒟0\mathcal{D}^{0};

9            
Allreduce: Aggregate attention outputs;

10            
FFN: Execute layernorm and FFN;

11            
Allreduce: Aggregate FFN outputs;

12

13       end foreach

14      Reduce: Sum final outputs with others;

15      
Postprocess: Execute layernorm, MLP, softmax, and sample next token;

16

17 end while

Algorithm 1 Master (with rank 0):

1
Download sliced weight files from the master node;

2
Start memory scheduler;

3
while *generated sequence not finished* do

4      
Broadcast: Receive embeddings, causal mask, and cache position from master;

5      
foreach *decoder layer l𝑙l* do

6            
Attention: Execute layernorm, self-attention, and store (Klk,Vlk)superscriptsubscript𝐾𝑙𝑘superscriptsubscript𝑉𝑙𝑘(K\_{l}^{k},V\_{l}^{k}) in KVCache 𝒟ksuperscript𝒟𝑘\mathcal{D}^{k};

7            
Allreduce: Aggregate attention outputs;

8            
FFN: Execute layernorm and FFN;

9            
Allreduce: Aggregate FFN outputs;

10

11       end foreach

12      Reduce: Send final output to master;

13

14 end while

Algorithm 2 Worker (with rank k𝑘k):

### 3.2 Allreduce latency analysis

Given the dynamic and heterogeneous nature of edge networks, we tested NetStorm (Li et al., [2024](#bib.bib13)) as the communication backend, but unfortunately, it resulted in high token latency. After further validation, we confirmed that this latency was not due to network bandwidth, but due to link latency.

To analyze the impact of network bandwidth and link latency, we make the following assumption.

###### Assumption 1.

Assume that the edge network adopts a physical topology as shown in Appendix [A.7](#A1.SS7 "A.7 Klonet testbed ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), the network links have the same latency τ𝜏\tau, the allreduce algorithm follows a tree-based structure of depth 2 for aggregation, and each device has the same computing power.

The allreduce latency can be expressed as t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎=2​L​(t𝚍𝚊𝚝𝚊+t𝚕𝚒𝚗𝚔+t𝚋𝚊𝚛𝚛𝚒𝚎𝚛+t𝚊𝚐𝚐𝚛)subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎2𝐿subscript𝑡𝚍𝚊𝚝𝚊subscript𝑡𝚕𝚒𝚗𝚔subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛subscript𝑡𝚊𝚐𝚐𝚛t\_{\mathtt{all\\_reduce}}=2L(t\_{\mathtt{data}}+t\_{\mathtt{link}}+t\_{\mathtt{barrier}}+t\_{\mathtt{aggr}}), where L𝐿L is the number of Transformer layers, t𝚍𝚊𝚝𝚊subscript𝑡𝚍𝚊𝚝𝚊t\_{\mathtt{data}} is the cumulative data transfer latency, t𝚕𝚒𝚗𝚔subscript𝑡𝚕𝚒𝚗𝚔t\_{\mathtt{link}} is the cumulative link latency, t𝚋𝚊𝚛𝚛𝚒𝚎𝚛subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛t\_{\mathtt{barrier}} is the cumulative barrier latency during aggregation, and t𝚊𝚐𝚐𝚛subscript𝑡𝚊𝚐𝚐𝚛t\_{\mathtt{aggr}} is the cumulative latency for aggregation calculation. Here we ignore t𝚊𝚐𝚐𝚛subscript𝑡𝚊𝚐𝚐𝚛t\_{\mathtt{aggr}} as it takes only 0.1 ms and thus negligible compared to other factors.

###### Proposition 1.

The bottleneck in allreduce is not network bandwidth, but link latency.

###### Proof.

The data transfer latency t𝚍𝚊𝚝𝚊=2​∑{i→j}∈𝒫h32​|𝑯|Bi​jsubscript𝑡𝚍𝚊𝚝𝚊2subscript→𝑖𝑗subscript𝒫ℎ32𝑯subscript𝐵𝑖𝑗t\_{\mathtt{data}}=2\sum\_{\{i\rightarrow j\}\in\mathcal{P}\_{h}}\frac{32|{\bm{H}}|}{B\_{ij}} depends on the size 32​|𝑯|32𝑯32|{\bm{H}}| of the data being transmitted and the bandwidth Bi​jsubscript𝐵𝑖𝑗B\_{ij} of the links in the path 𝒫hsubscript𝒫ℎ\mathcal{P}\_{h}, here 𝒫hsubscript𝒫ℎ\mathcal{P}\_{h} is an index sequence from device hℎh to the master device. For example, in the case of Llama 2-70B with a hidden size |𝑯|=8192𝑯8192|{\bm{H}}|=8192 and a network bandwidth of 300 Mbps, the data transfer latency is only t𝚍𝚊𝚝𝚊=3.4subscript𝑡𝚍𝚊𝚝𝚊3.4t\_{\mathtt{data}}=3.4 ms, which is negligible compared to other latencies. In addition, experiment results in Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Scaling over varying edge conditions ‣ 4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") show that increasing the network bandwidth does not significantly reduce token latency, further confirming that data transfer and network bandwidth is not the bottleneck.

The link latency t𝚕𝚒𝚗𝚔subscript𝑡𝚕𝚒𝚗𝚔t\_{\mathtt{link}}, which is often neglected, emerges as the main issue. For example, the path from device h2subscriptℎ2h\_{2} to h1subscriptℎ1h\_{1} via h8subscriptℎ8h\_{8} follows the route h2→r2→r9→r8→h8→r8→r9→r1→h1→subscriptℎ2subscript𝑟2→subscript𝑟9→subscript𝑟8→subscriptℎ8→subscript𝑟8→subscript𝑟9→subscript𝑟1→subscriptℎ1h\_{2}\rightarrow r\_{2}\rightarrow r\_{9}\rightarrow r\_{8}\rightarrow h\_{8}\rightarrow r\_{8}\rightarrow r\_{9}\rightarrow r\_{1}\rightarrow h\_{1}, resulting in a total link latency of 16​τ16𝜏16\tau, where τ𝜏\tau is the per-hop link latency. To isolate the impact of

link latency, we ran allreduce with only 4 bytes of data, excluding data transfer t𝚍𝚊𝚝𝚊subscript𝑡𝚍𝚊𝚝𝚊t\_{\mathtt{data}} and barrier latencies t𝚋𝚊𝚛𝚛𝚒𝚎𝚛subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛t\_{\mathtt{barrier}}. The results, shown in Figure [3](#S3.F3.9 "Figure 3 ‣ Proof. ‣ 3.2 Allreduce latency analysis ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), demonstrate that the per-link latency τ𝜏\tau significantly impacts allreduce latency. This indicates that an inefficient allreduce algorithm, where multiple hops are required (e.g., ring (Ye et al., [2024](#bib.bib22); Shoeybi et al., [2019](#bib.bib18)) or tree-based (Zhou et al., [2021](#bib.bib24); Li et al., [2024](#bib.bib13)) algorithms), will further amplifies this impact. For example, with the ring algorithm, allreduce requires 7 communication steps for reducescatter and 7 for allgather, resulting in a total link latency of 56​τ56𝜏56\tau, which is 3.5×\times higher than the tree-based setup.

The barrier latency, t𝚋𝚊𝚛𝚛𝚒𝚎𝚛subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛t\_{\mathtt{barrier}}, arises from synchronization

Figure 3: Impact of link latency τ𝜏\tau.

delays during data aggregation. Given the assumption that all devices have equal computing power and network links have equal latencies, the barrier latency can be approximated as negligible:

|  |  |  |  |
| --- | --- | --- | --- |
|  | t𝚋𝚊𝚛𝚛𝚒𝚎𝚛=max⁡{∑(i→j)∈𝒫32​|𝑯|Bi​j,∀𝒫}−min⁡{∑(i→j)∈𝒫32​|𝑯|Bi​j,∀𝒫}≈0.subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛subscript→𝑖𝑗𝒫32𝑯subscript𝐵𝑖𝑗for-all𝒫subscript→𝑖𝑗𝒫32𝑯subscript𝐵𝑖𝑗for-all𝒫0t\_{\mathtt{barrier}}=\max\{\sum\_{(i\rightarrow j)\in\mathcal{P}}\frac{32|{\bm{H}}|}{B\_{ij}},\forall\mathcal{P}\}-\min\{\sum\_{(i\rightarrow j)\in\mathcal{P}}\frac{32|{\bm{H}}|}{B\_{ij}},\forall\mathcal{P}\}\approx 0. |  | (3) |

Thus, link latency t𝚕𝚒𝚗𝚔subscript𝑡𝚕𝚒𝚗𝚔t\_{\mathtt{link}} emerges as the key factor in allreduce latency.
∎

###### Proposition 2.

The star-based allreduce is more effective for TPI-LLM in high-latency networks.

Despite past criticism, the star-based allreduce, where workers push data directly to the master for aggregation and pull the result back (Chen et al., [2015](#bib.bib4)), stands out as the best choice (see Appendix [A.1](#A1.SS1 "A.1 Proof of proposition 2 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") for a detailed proof). It has minimal hops (888), lowest link latency (8​τ8𝜏8\tau), zero intermediate barriers, and avoids the single-point issue due to the small data size (256 KB per device), making it the preferred allreduce algorithm for TPI-LLM.

### 3.3 Sliding Window Memory Scheduling

Quantizations like FP16 and INT8 are common for NVIDIA GPUs with CUDA support, but most edge devices lack CUDA and prefer full precision for faster computation due to their general-purpose CPU design. As a result, while tensor parallelism helps distribute memory costs across devices, the memory load remains high. Thus, memory scheduling is still required to manage these loads.

We introduce a memory scheduler, which manages memory by dynamically loading and unloading model weights during inference, ensuring that only the necessary parts are kept in memory (see Appendix [A.2](#A1.SS2 "A.2 A simple-to-use memory scheduler ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") for potential use). The memory scheduler operates on a daemon thread to asynchronously handle memory operations. To maintain the peak memory footprint, it uses a sliding window and preloads weights for upcoming layers while unloading those that have been processed.

As mentioned in Section [3.1](#S3.SS1 "3.1 The Parallel Framework Design of TPI-LLM System ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), each Transformer layer is divided into attention computing, allreduce, FFN computing, and allreduce. For simplicity, in Figure [4](#S3.F4 "Figure 4 ‣ 3.3 Sliding Window Memory Scheduling ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), we assume the delays for these stages and weight loading to be equal. In each time slot, the memory scheduler asynchronously loads weights for either an attention or FFN block. By overlapping weight loading with ongoing computations and communications, it hides the I/O latency associated with loading weights from disk. For example, in Figure [4](#S3.F4 "Figure 4 ‣ 3.3 Sliding Window Memory Scheduling ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), the memory scheduler loads one more block during each allreduce until the sliding window reaches its size. As computations and communications proceed, we ensure weights are always ready when needed, allowing for seamless inference without computational stalls.

Figure 4: An illustration of the sliding window memory scheduling. Blue blocks indicate the blocks currently executed, with numbered blocks for attention or FFN computing and unnumbered blocks for allreduce communication. Green blocks indicate loaded model weights. The dashed box represents the sliding window, with size 4 in this case.

Next, we provide the conditions for this mechanism to reach a steady state, under which all required weights are loaded before computation starts.

###### Proposition 3 (Loose Steady Condition).

The memory scheduler reaches a steady state when the following condition is met:

|  |  |  |  |
| --- | --- | --- | --- |
|  | t𝚊𝚝𝚝𝚗+t𝚏𝚏𝚗+2​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎≥τ𝚏𝚏𝚗+τ𝚊𝚝𝚝𝚗,subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗2subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗subscript𝜏𝚊𝚝𝚝𝚗t\_{\mathtt{attn}}+t\_{\mathtt{ffn}}+2t\_{\mathtt{all\\_reduce}}\geq\tau\_{\mathtt{ffn}}+\tau\_{\mathtt{attn}}, |  | (4) |

and one of the following conditions is met:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+(l−1)⋅t𝚏𝚏𝚗+(2​l−1)⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙1subscript𝑡𝚏𝚏𝚗⋅2𝑙1subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle l\cdot t\_{\mathtt{attn}}+(l-1)\cdot t\_{\mathtt{ffn}}+(2l-1)\cdot t\_{\mathtt{all\\_reduce}} | ≥l⋅τ𝚏𝚏𝚗+(l−1)⋅τ𝚊𝚝𝚝𝚗,∀l∈{1,⋯,L},formulae-sequenceabsent⋅𝑙subscript𝜏𝚏𝚏𝚗⋅𝑙1subscript𝜏𝚊𝚝𝚝𝚗for-all𝑙1⋯𝐿\displaystyle\geq l\cdot\tau\_{\mathtt{ffn}}+(l-1)\cdot\tau\_{\mathtt{attn}},~{}\forall l\in\{1,\cdots,L\}, |  | (5) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (l−1)⋅t𝚊𝚝𝚝𝚗+l⋅t𝚏𝚏𝚗+(2​l−1)⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙1subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙subscript𝑡𝚏𝚏𝚗⋅2𝑙1subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle(l-1)\cdot t\_{\mathtt{attn}}+l\cdot t\_{\mathtt{ffn}}+(2l-1)\cdot t\_{\mathtt{all\\_reduce}} | ≥(l−1)⋅τ𝚏𝚏𝚗+l⋅τ𝚊𝚝𝚝𝚗,∀l∈{1,⋯,L},formulae-sequenceabsent⋅𝑙1subscript𝜏𝚏𝚏𝚗⋅𝑙subscript𝜏𝚊𝚝𝚝𝚗for-all𝑙1⋯𝐿\displaystyle\geq(l-1)\cdot\tau\_{\mathtt{ffn}}+l\cdot\tau\_{\mathtt{attn}},~{}\forall l\in\{1,\cdots,L\}, |  | (6) |

where t𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚝𝚝𝚗t\_{\mathtt{attn}} and t𝚏𝚏𝚗subscript𝑡𝚏𝚏𝚗t\_{\mathtt{ffn}} are times required for attention and FFN computation, t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎t\_{\mathtt{all\\_reduce}} is the allreduce latency, τ𝚏𝚏𝚗subscript𝜏𝚏𝚏𝚗\tau\_{\mathtt{ffn}} and τ𝚊𝚝𝚝𝚗subscript𝜏𝚊𝚝𝚝𝚗\tau\_{\mathtt{attn}} are times required to load attention and FFN weights, and L𝐿L is the number of Transformer layers.

This condition is loose but a bit hard to assess, so we present a tighter, more intuitive condition.

###### Proposition 4 (Tight Steady Condition).

t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎≥τ𝚏𝚏𝚗subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}\geq\tau\_{\mathtt{ffn}} and t𝚏𝚏𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎≥τ𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗t\_{\mathtt{ffn}}+t\_{\mathtt{all\\_reduce}}\geq\tau\_{\mathtt{attn}}.

The proofs can be found in Appendices [A.3](#A1.SS3 "A.3 Proof of proposition 3 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") and [A.4](#A1.SS4 "A.4 Proof of proposition 4 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"). This conclusion is straightforward. If the previous block’s computation and allreduce time cover the current block’s weight loading time, the memory scheduler can fully hide the disk I/O latency. As an example, in Section [4.4](#S4.SS4 "4.4 Real case study ‣ 4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), we use 4 laptops with Llama 2-7B, setting pi=0.25subscript𝑝𝑖0.25p\_{i}=0.25 and w=4𝑤4w=4. We measured t𝚊𝚝𝚝𝚗=11subscript𝑡𝚊𝚝𝚝𝚗11t\_{\mathtt{attn}}=11 ms, t𝚏𝚏𝚗=17subscript𝑡𝚏𝚏𝚗17t\_{\mathtt{ffn}}=17 ms, t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎=14subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎14t\_{\mathtt{all\\_reduce}}=14 ms, τ𝚊𝚝𝚝𝚗=18subscript𝜏𝚊𝚝𝚝𝚗18\tau\_{\mathtt{attn}}=18 ms, and τ𝚏𝚏𝚗=30subscript𝜏𝚏𝚏𝚗30\tau\_{\mathtt{ffn}}=30 ms. While the tight steady condition is not met, the loose steady condition is met, allowing the memory scheduler to achieve steady state.

###### Proposition 5 (Peak Memory Footprint).

If the memory scheduler reaches a steady state, the peak memory footprint of the master and worker can be expressed as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M𝚖𝚊𝚜𝚝𝚎𝚛subscript𝑀𝚖𝚊𝚜𝚝𝚎𝚛\displaystyle M\_{\mathtt{master}} | =γ×{h​v+h,if ​w=12​h​v+h,if ​w=22​h​v+h+⌊w−22⌋​(2​(1+ba)​h2​pi+h)+⌊w−12⌋​(3​h​s​pi+h),if ​w≥3absent𝛾casesℎ𝑣ℎif 𝑤12ℎ𝑣ℎif 𝑤22ℎ𝑣ℎ𝑤2221𝑏𝑎superscriptℎ2subscript𝑝𝑖ℎ𝑤123ℎ𝑠subscript𝑝𝑖ℎif 𝑤3\displaystyle=\gamma\times\begin{cases}hv+h,&\text{if }w=1\\ 2hv+h,&\text{if }w=2\\ 2hv+h+\left\lfloor\frac{w-2}{2}\right\rfloor\left(2(1+\frac{b}{a})h^{2}p\_{i}+h\right)+\left\lfloor\frac{w-1}{2}\right\rfloor(3hsp\_{i}+h),&\text{if }w\geq 3\end{cases} |  | (7) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | M𝚠𝚘𝚛𝚔𝚎𝚛subscript𝑀𝚠𝚘𝚛𝚔𝚎𝚛\displaystyle M\_{\mathtt{worker}} | =γ×(⌊w2⌋​(2​(1+ba)​h2​pi+h)+⌊w+12⌋​(3​h​s​pi+h)),absent𝛾𝑤221𝑏𝑎superscriptℎ2subscript𝑝𝑖ℎ𝑤123ℎ𝑠subscript𝑝𝑖ℎ\displaystyle=\gamma\times\left(\left\lfloor\frac{w}{2}\right\rfloor(2(1+\frac{b}{a})h^{2}p\_{i}+h)+\left\lfloor\frac{w+1}{2}\right\rfloor(3hsp\_{i}+h)\right), |  | (8) |

where hℎh is the hidden size, v𝑣v is the vocabulary size, a𝑎a is the number of attention heads, b𝑏b is the number of key-value heads, s𝑠s is the intermediate size, pisubscript𝑝𝑖p\_{i} is the proportion of parameters handled by device i𝑖i, w𝑤w is the memory window size, and γ𝛾\gamma is a memory scaling factor.

The proof can be found in Appendix [A.5](#A1.SS5 "A.5 Proof of proposition 5 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"). However, if a slow disk I/O disrupts the steady state, the memory scheduler will retain some FFN blocks in memory to reduce disk access frequency.

###### Proposition 6 (Loose Steady Condition with Block Retention).

Let the memory scheduler retain one FFN block in memory every T𝑇T FFN blocks, the condition to reach a steady state is then

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+l⋅t𝚏𝚏𝚗+2​l⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙subscript𝑡𝚏𝚏𝚗⋅2𝑙subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle l\cdot t\_{\mathtt{attn}}+l\cdot t\_{\mathtt{ffn}}+2l\cdot t\_{\mathtt{all\\_reduce}} | ≥(l−⌈lT⌉)⋅τ𝚏𝚏𝚗+l⋅τ𝚊𝚝𝚝𝚗,absent⋅𝑙𝑙𝑇subscript𝜏𝚏𝚏𝚗⋅𝑙subscript𝜏𝚊𝚝𝚝𝚗\displaystyle\geq(l-\left\lceil\frac{l}{T}\right\rceil)\cdot\tau\_{\mathtt{ffn}}+l\cdot\tau\_{\mathtt{attn}}, |  | (9) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+(l−1)⋅t𝚏𝚏𝚗+(2​l−1)⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙1subscript𝑡𝚏𝚏𝚗⋅2𝑙1subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle l\cdot t\_{\mathtt{attn}}+(l-1)\cdot t\_{\mathtt{ffn}}+(2l-1)\cdot t\_{\mathtt{all\\_reduce}} | ≥(l−⌈lT⌉)⋅τ𝚏𝚏𝚗+(l−1)⋅τ𝚊𝚝𝚝𝚗.absent⋅𝑙𝑙𝑇subscript𝜏𝚏𝚏𝚗⋅𝑙1subscript𝜏𝚊𝚝𝚝𝚗\displaystyle\geq(l-\left\lceil\frac{l}{T}\right\rceil)\cdot\tau\_{\mathtt{ffn}}+(l-1)\cdot\tau\_{\mathtt{attn}}. |  | (10) |

The proof can be found in Appendix [A.6](#A1.SS6 "A.6 Proof of proposition 6 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"). By setting an appropriate T𝑇T, idle memory can help the scheduler reach a steady state, thus achieving a tradeoff between memory use and inference speed.

## 4 Experiments

Prototype and Testbed. We implemented the prototype of TPI-LLM111Open available at: <https://anonymous.4open.science/r/tpi-llm>. with 3K LoC using PyTorch and Transformers to provide flexible support for various sizes and versions of pretrained LLMs. Our testbed, illustrated in Appendix [A.7](#A1.SS7 "A.7 Klonet testbed ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), was built upon Klonet (Ma et al., [2024](#bib.bib14)) to create an edge network environment, emulating realistic conditions with configurable properties like network topology, bandwidth, and latency. By default, 8 edge devices were emulated on 2 Intel Xeon Gold 5220R CPUs, each limited to 8 logical cores, 8 GB of memory, and 4 GB of swap. Network bandwidth between devices was set to 300 Mbps with a 1 ms latency.

Models. The inference speed of TPI-LLM is significantly affected by the model architecture. Deeper layers, more parameters, larger hidden sizes, and more attention heads increase the computational latency. Additionally, deeper layers result in more allreduce communications, and a larger hidden size leads to greater traffic. We tested with various models of different sizes, including Llama 2-3B/7B/13B/70B, Llama 3-8B/70B, and Yi-34B. See Appendix [A.8](#A1.SS8 "A.8 Configurations of the used models ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") for their configuration details.

### 4.1 Overview of tpi-llm performance

Fit 70B LLMs into edge devices and run in high efficiency. We tested the performance of TPI-LLM with a focus on 3 key metrics: time-to-first-token (TTFT), token latency, and peak memory footprint per device. The memory window size is set to 2 by default. As shown in Table [1](#S4.T1 "Table 1 ‣ 4.1 Overview of tpi-llm performance ‣ 4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), without the memory scheduler, the full weights are loaded into the memory at once. Despite that these weights have been distributed across multiple devices, the memory is still insufficient for larger models like Yi-34B and Llama 2/3/3.1-70B. Instead, enabling our memory scheduler significantly reduces the peak memory footprint, allowing larger models to run efficiently. For example, the Llama 2-70B model requires just 3.1 GB of memory per device, and the Llama 3.1-70B model fits within device limits. The results are summarized in Table [1](#S4.T1 "Table 1 ‣ 4.1 Overview of tpi-llm performance ‣ 4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices").

Table 1: The TTFT, token latency, and peak memory footprint per device of TPI-LLM.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model (FP32) | Memory Scheduler Disabled | | | Memory Scheduler Enabled | | |
| TTFT | Latency | Memory | TTFT | Latency | Memory |
| Llama 2-3B | 2.3 s | 1.0 s/token | 2.8 GB | 2.0 s | 1.9 s/token | 1.4 GB |
| Llama 2-7B | 3.1 s | 1.2 s/token | 4.5 GB | 3.0 s | 2.6 s/token | 1.7 GB |
| Llama 2-13B | 5.1 s | 1.9 s/token | 8.1 GB | 5.8 s | 2.9 s/token | 2.1 GB |
| Llama 2-70B | OOM | OOM | 34.9 GB | 29.4 s | 26.1 s/token | 3.1 GB |
| Llama 3.1-8B | 4.5 s | 1.5 s/token | 8.5 GB | 4.5 s | 4.3 s/token | 5.4 GB |
| Llama 3.1-70B | OOM | OOM | 42.3 GB | 32.9 s | 29.9 s/token | 11.3 GB |
| Yi-34B | OOM | OOM | 20.4 GB | 15.7 s | 13.7 s/token | 4.9 GB |




Table 2: Peak memory footprint per device with the memory window size set to 2.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Memory Scheduler Disabled (GB) | | | | Memory Scheduler Enabled (GB) | | | |
| Model (FP32) | N=2𝑁2N=2 | N=4𝑁4N=4 | N=6𝑁6N=6 | N=8𝑁8N=8 | N=2𝑁2N=2 | N=4𝑁4N=4 | N=6𝑁6N=6 | N=8𝑁8N=8 |
| Llama 2-3B | 7.3 | 4.3 | 3.2 | 2.8 | 1.5 | 1.4 | 1.4 | 1.4 |
| Llama 2-7B | 13.7 | 7.7 | 5.5 | 4.5 | 2.0 | 1.8 | 1.7 | 1.7 |
| Llama 2-13B | 25.7 | 13.9 | 9.8 | 8.1 | 2.3 | 2.2 | 2.2 | 2.1 |
| Llama 2-70B | 130 | 66.6 | 46.6 | 34.9 | 3.7 | 3.3 | 3.3 | 3.1 |
| Llama 3.1-8B | 18.4 | 11.8 | 9.4 | 8.5 | 5.9 | 5.6 | 5.5 | 5.4 |
| Llama 3.1-70B | 137.7 | 74.0 | 51.1 | 42.3 | 10.8 | 10.5 | 11.4 | 11.3 |
| Yi-34B | 67 | 36.4 | 23.9 | 20.4 | 5.0 | 5.0 | 5.0 | 4.9 |

No need for dozens of devices, one or two are enough to run 70B models. We used 8 devices by default, but can fewer devices run 70B-scale models? Table [2](#S4.T2 "Table 2 ‣ 4.1 Overview of tpi-llm performance ‣ 4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") gives detailed peak memory footprints with varying number of devices. Without the memory scheduler, full weights are loaded onto the devices, and with fewer devices, the memory load increases. For instance, using only 2 devices limits users to smaller models, like those between 3B and 7B. However, with the memory scheduler enabled, only a few layers’ weights are loaded and distributed across devices. This allows even larger models, such as 70B, to run smoothly on just 2 devices. Appendix [A.9](#A1.SS9 "A.9 Peak memory footprint with memory window size 4 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") shows the case with a memory window size of 4, which requires slightly more memory but faster speed. The peak memory footprint in TPI-LLM is primarily determined by the product of vocabulary size and hidden size, which is detailed in equation ([7](#S3.E7 "In Proposition 5 (Peak Memory Footprint). ‣ 3.3 Sliding Window Memory Scheduling ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices")) and can be further reduced in our future work.

### 4.2 Scaling over varying edge conditions

Computation remains the bottleneck, not network bandwidth. In this experiment, we examined the token latency of TPI-LLM under different edge conditions, the results are shown in Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Scaling over varying edge conditions ‣ 4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"). As expected, increasing the number of devices reduces the computing load on each device, significantly lowering token latency, and more CPU cores also contribute to a reduced latency. Instead, a limited network bandwidth was no longer a bottleneck, boosting it from 300 Mbps to 1 Gbps had little effect on latency due to the tiny data size (only 256 KB) during each allreduce. Thus, the main bottleneck remains in the computation, which our future work should focus on.

Figure 5: Token latency over varying number of devices, CPU cores, and network bandwidth on Llama 2-70B.

### 4.3 Comparison with benchmarks

Figure 6: Comparison of TPI-LLM with three benchmarks.

We compared the TPI-LLM with 3 benchmarks: (a) Standalone: LLM inference is executed only on a single edge device using Transformers (Wolf et al., [2020](#bib.bib21)). (b) Model Parallelism (MP): Since only one user is served at a time, the pipeline parallelism (Zhang et al., [2024](#bib.bib23); Mei et al., [2024](#bib.bib15); Borzunov et al., [2024](#bib.bib3)) degrades to the model parallelism, where different layer sequences are distributed across multiple devices. Each device computes its layers and passes the result to the next device until the entire inference is complete. (c) Galaxy (Ye et al., [2024](#bib.bib22)) combines tensor and sequence parallelism and overlaps communication and computation to accelerate inference. They all run in FP32 mode.

Run larger models with lower latency and memory usage. As shown in Figure [6](#S4.F6 "Figure 6 ‣ 4.3 Comparison with benchmarks ‣ 4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), a limited memory on a single device makes it challenging to run even 3B models in a standalone mode. MP addresses this by the collaboration of 8 devices, allowing models up to 13B, but suffers from high latency due to pipeline bubbles. Galaxy tries to reduce such latency by combining tensor and sequence parallelism. However, in Section [3.2](#S3.SS2 "3.2 Allreduce latency analysis ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), we concluded that the network bandwidth was no longer the issue, and the real problem is the link latency. Galaxy’s use of a ring algorithm for reducescatter and allgather forces each link to be traversed at least 14 times. This causes high link latency and outweighs the benefits of parallel computing, ultimately resulting in a higher token latency than MP. In contrast, TPI-LLM adopts a star-based allreduce algorithm, minimizing hops and cumulative link latency. Combined with the blocking-free memory scheduler, TPI-LLM delivers significantly lower token latency and memory footprint, even with larger 70B models.

### 4.4 Real case study

In this study, we used 4 laptops with different CPU architectures and memory capacities, connected via a local Wi-Fi router. The testbed and configurations are detailed in Appendix [A.10](#A1.SS10 "A.10 Real testbed and configurations ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"). Macbook Pro was used by default. Due to the lack of CUDA, all computations were performed in full precision. As shown in Table [3](#S4.T3 "Table 3 ‣ 4.4 Real case study ‣ 4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), Transformers loaded the entire model into the CPU memory, and when memory was insufficient, the operating system offloaded data to the swap. This frequent swap exchange significantly increased TTFT and token latency, even for smaller 3B models. As the model size grows, the swap space overflowed, finally leading to OOM errors. As a more efficient alternative, Accelerate (Gugger et al., [2022](#bib.bib7)) instantly loads layer weights only when required for the computation and reduces unnecessary data I/O. While it speeds up inference, due to implementation flaws on disk offloading, it still requires loading full weights before splitting and offloading them to disk. This results in OOM errors when the model size reaches 13B.

TPI-LLM stands out in TTFT, token latency, and model size. Our memory scheduler (Transformers+MS) outperforms Transformers and Accelerate in both TTFT and token latency across all model sizes. This is because our memory scheduler employs a sliding window mechanism, where a daemon thread asynchronously preloads the weights needed for upcoming computations. By overlapping data I/O with computations and communications, the scheduler avoids delays caused by disk I/O blocks, ensuring smoother and faster inference. To further speed up inference, we integrate the computing power of 4 laptops to serve TPI-LLM. By distributing the computational load across 4 laptops, the reduction in computing time far exceeds communication delays, so both TTFT and token latency are further reduced. The results from using 3 laptops are shown in Appendix [A.11](#A1.SS11 "A.11 Case study with 3 laptops ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), indicating a slightly higher latency due to reduced parallelism.

Table 3: Comparison of Transformers, Accelerate, Transformers+MS, and TPI-LLM on 4 laptops.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model (FP32) | Transformers | | Accelerate | | Transformers + MS | | TPI-LLM | |
| TTFT  (s) | Latency  (s/token) | TTFT  (s) | Latency  (s/token) | TTFT  (s) | Latency  (s/token) | TTFT  (s) | Latency  (s/token) |
| Llama 2-3B | 61 | 30 | 24 | 16 | 4 | 3 | 2.5 | 2 |
| Llama 2-7B | 115 | 56 | 30 | 26 | 13 | 8 | 6 | 5 |
| Llama 2-13B | OOM | OOM | OOM | OOM | 22 | 18 | 10 | 9 |
| Llama 3.1-8B | 133 | 65 | 37 | 31 | 20 | 12 | 11 | 8 |
| Yi-34B | OOM | OOM | OOM | OOM | 185 | 55 | 33 | 29 |

## 5 Conclusion

In this paper, we concluded that tensor parallelism can be more effective than pipeline parallelism on low-resource devices, and presented a compute- and memory-efficient tensor parallel inference system, named TPI-LLM, to serve 70B-scale LLMs. TPI-LLM is designed with user prompt and generated sequence privacy in mind, by keeping sensitive raw data local in the users’ devices. It leverages a sliding window memory scheduler to dynamically manage layer weights during inference with disk I/O latency overlapped by onging computations and communications, allowing larger models to run smoothly on devices with very limited memory. Our analysis showed that link latency, not bandwidth, emerges as the main issue, so TPI-LLM implements a star-based allreduce algorithm, rather than the commonly used ring- and tree-based algorithms. Through extensive experiments on emulated and real testbeds, TPI-LLM demonstrated significantly lower TTFT, token latency, and peak memory footprint compared to Transformers, Accelerate, Galaxy, and enabled serving larger-scale LLMs such as Yi-34B and Llama 2/3/3.1-70B on low-memory devices.

## Reproducibility

We have made efforts to ensure reproducibility by providing the source code at <https://anonymous.4open.science/r/tpi-llm>, with a detailed README for guidance included. To ease the use, a prebuilt Docker image is also provided. Key experimental setups are given in Section [4](#S4 "4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") of the paper.

## References

* Agrawal et al. (2024)

  Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee.
  Taming Throughput-Latency tradeoff in LLM inference with Sarathi-Serve.
  In *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, pp.  117–134, Santa Clara, CA, July 2024. USENIX Association.
  ISBN 978-1-939133-40-3.
  URL <https://www.usenix.org/conference/osdi24/presentation/agrawal>.
* AI et al. (2024)

  01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai.
  Yi: Open foundation models by 01.ai, 2024.
* Borzunov et al. (2024)

  Alexander Borzunov, Max Ryabinin, Artem Chumachenko, Dmitry Baranchuk, Tim Dettmers, Younes Belkada, Pavel Samygin, and Colin A Raffel.
  Distributed inference and fine-tuning of large language models over the internet.
  *Advances in Neural Information Processing Systems*, 36, 2024.
* Chen et al. (2015)

  Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang.
  Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems.
  *arXiv preprint arXiv:1512.01274*, 2015.
* Dubey et al. (2024)

  Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al.
  The llama 3 herd of models.
  *arXiv preprint arXiv:2407.21783*, 2024.
* Elfwing et al. (2018)

  Stefan Elfwing, Eiji Uchibe, and Kenji Doya.
  Sigmoid-weighted linear units for neural network function approximation in reinforcement learning.
  *Neural networks*, 107:3–11, 2018.
* Gugger et al. (2022)

  Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan.
  Accelerate: Training and inference at scale made simple, efficient and adaptable.
  <https://github.com/huggingface/accelerate>, 2022.
* Jin et al. (2023)

  Yunho Jin, Chun-Feng Wu, David Brooks, and Gu-Yeon Wei.
  s3superscript𝑠3s^{3}: Increasing gpu utilization during generative inference for higher throughput.
  *Advances in Neural Information Processing Systems*, 36:18015–18027, 2023.
* Kwon et al. (2023)

  Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica.
  Efficient memory management for large language model serving with pagedattention.
  In *Proceedings of the 29th Symposium on Operating Systems Principles*, pp.  611–626, 2023.
* Lee et al. (2024)

  Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim.
  Infinigen: Efficient generative inference of large language models with dynamic kv cache management.
  In *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, pp.  155–172, 2024.
* Li et al. (2014)

  Mu Li, David G Andersen, Jun Woo Park, Alexander J Smola, Amr Ahmed, Vanja Josifovski, James Long, Eugene J Shekita, and Bor-Yiing Su.
  Scaling distributed machine learning with the parameter server.
  In *11th USENIX Symposium on operating systems design and implementation (OSDI 14)*, pp.  583–598, 2014.
* Li et al. (2023)

  Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
  AlpaServe: Statistical multiplexing with model parallelism for deep learning serving.
  In *17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)*, pp.  663–679, Boston, MA, July 2023. USENIX Association.
  ISBN 978-1-939133-34-2.
  URL <https://www.usenix.org/conference/osdi23/presentation/li-zhouhan>.
* Li et al. (2024)

  Zonghang Li, Wenjiao Feng, Weibo Cai, Hongfang Yu, Long Luo, Gang Sun, Hongyang Du, and Dusit Niyato.
  Accelerating geo-distributed machine learning with network-aware adaptive tree and auxiliary route.
  *IEEE/ACM Transactions on Networking*, 2024.
* Ma et al. (2024)

  Tie Ma, Long Luo, Hongfang Yu, Xi Chen, Jingzhao Xie, Chongxi Ma, Yunhan Xie, Gang Sun, Tianxi Wei, Li Chen, et al.
  Klonet: an easy-to-use and scalable platform for computer networks education.
  In *21st USENIX Symposium on Networked Systems Design and Implementation*, pp.  2025–2046, 2024.
* Mei et al. (2024)

  Yixuan Mei, Yonghao Zhuang, Xupeng Miao, Juncheng Yang, Zhihao Jia, and Rashmi Vinayak.
  Helix: Distributed serving of large language models via max-flow on heterogeneous gpus.
  *arXiv preprint arXiv:2406.01566*, 2024.
* Miao et al. (2024)

  Xupeng Miao, Chunan Shi, Jiangfei Duan, Xiaoli Xi, Dahua Lin, Bin Cui, and Zhihao Jia.
  Spotserve: Serving generative large language models on preemptible instances.
  In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, pp.  1112–1127, 2024.
* Rasley et al. (2020)

  Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He.
  Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.
  In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pp.  3505–3506, 2020.
* Shoeybi et al. (2019)

  Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro.
  Megatron-lm: Training multi-billion parameter language models using model parallelism.
  *arXiv preprint arXiv:1909.08053*, 2019.
* Touvron et al. (2023)

  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
  Llama 2: Open foundation and fine-tuned chat models.
  *arXiv preprint arXiv:2307.09288*, 2023.
* Wei et al. (2024)

  Yuanxin Wei, Shengyuan Ye, Jiazhi Jiang, Xu Chen, Dan Huang, Jiangsu Du, and Yutong Lu.
  Communication-efficient model parallelism for distributed in-situ transformer inference.
  In *2024 Design, Automation & Test in Europe Conference & Exhibition*, pp.  1–6. IEEE, 2024.
* Wolf et al. (2020)

  Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al.
  Transformers: State-of-the-art natural language processing.
  In *Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations*, pp.  38–45, 2020.
* Ye et al. (2024)

  Shengyuan Ye, Jiangsu Du, Liekang Zeng, Wenzhong Ou, Xiaowen Chu, Yutong Lu, and Xu Chen.
  Galaxy: A resource-efficient collaborative edge ai system for in-situ transformer inference.
  *arXiv preprint arXiv:2405.17245*, 2024.
* Zhang et al. (2024)

  Mingjin Zhang, Jiannong Cao, Xiaoming Shen, and Zeyang Cui.
  Edgeshard: Efficient llm inference via collaborative edge computing.
  *arXiv preprint arXiv:2405.14371*, 2024.
* Zhou et al. (2021)

  Huaman Zhou, Weibo Cai, Zonghang Li, Hongfang Yu, Ling Liu, Long Luo, and Gang Sun.
  Tsengine: Enable efficient communication overlay in distributed machine learning in wans.
  *IEEE Transactions on Network and Service Management*, 18(4):4846–4859, 2021.

## Appendix A Appendix

### A.1 Proof of proposition 2

In conventional data parallel systems, each device sends several gigabytes of data, putting significant pressure on network bandwidth. This makes data transfer latency a major concern, while link latency becomes negligible. Then, tree and ring-based algorithms are introduced to optimize the data transfer. However, they do not apply to our case. In TPI-LLM, each device only sends a small amount of data, usually just tens of kilobytes. This tiny data size does not strain the network, so data transfer latency is minimal. Instead, in edge networks where wireless communication causes higher transmission delays, link latency becomes more significant than data transfer latency. As a result, the commonly used tree and ring-based allreduce algorithms are less effective.

Let us consider 1 master and 2 workers connected via a router. In Figure [7](#A1.F7 "Figure 7 ‣ A.1 Proof of proposition 2 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), we compare the traffic models of star, tree, and ring-based algorithms. In star-based allreduce, worker 1 sends data directly to the master via the router, and the allreduce latency (includes reduce and broadcast) is t𝚜𝚝𝚊𝚛=2​(t𝚍𝚊𝚝𝚊+t𝚕𝚒𝚗𝚔)+t𝚋𝚊𝚛𝚛𝚒𝚎𝚛+t𝚊𝚐𝚐𝚛subscript𝑡𝚜𝚝𝚊𝚛2subscript𝑡𝚍𝚊𝚝𝚊subscript𝑡𝚕𝚒𝚗𝚔subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛subscript𝑡𝚊𝚐𝚐𝚛t\_{\mathtt{star}}=2(t\_{\mathtt{data}}+t\_{\mathtt{link}})+t\_{\mathtt{barrier}}+t\_{\mathtt{aggr}}. In this model, the router only forwards data packets.

Figure 7: Comparison of traffic models for star, tree, and ring-based allreduce algorithms.

In tree-based allreduce, data from worker 1 must first go through worker 2 before reaching the master, so there are 2 hops involved. In this process, worker 1 sends its data to worker 2, which aggregates it and forwards the result to the master. Once the global aggregation is complete, the final result is broadcast back to all workers. The total time for this process is t𝚝𝚛𝚎𝚎=3​t𝚍𝚊𝚝𝚊+4​t𝚕𝚒𝚗𝚔+2​t𝚋𝚊𝚛𝚛𝚒𝚎𝚛+2​t𝚊𝚐𝚐𝚛subscript𝑡𝚝𝚛𝚎𝚎3subscript𝑡𝚍𝚊𝚝𝚊4subscript𝑡𝚕𝚒𝚗𝚔2subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛2subscript𝑡𝚊𝚐𝚐𝚛t\_{\mathtt{tree}}=3t\_{\mathtt{data}}+4t\_{\mathtt{link}}+2t\_{\mathtt{barrier}}+2t\_{\mathtt{aggr}}.

In ring-based allreduce, each device communicates directly with its neighbors in a ring topology. Data is divided and sent in a sequence around the ring, with each device receiving, aggregating, and passing the data to the next device. Unlike star or tree-based methods, there is no central device, and data flows continuously between the devices. The total time for the ring-based allreduce is t𝚛𝚒𝚗𝚐=43​t𝚍𝚊𝚝𝚊+4​t𝚕𝚒𝚗𝚔+3​t𝚋𝚊𝚛𝚛𝚒𝚎𝚛+23​t𝚊𝚐𝚐𝚛subscript𝑡𝚛𝚒𝚗𝚐43subscript𝑡𝚍𝚊𝚝𝚊4subscript𝑡𝚕𝚒𝚗𝚔3subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛23subscript𝑡𝚊𝚐𝚐𝚛t\_{\mathtt{ring}}=\frac{4}{3}t\_{\mathtt{data}}+4t\_{\mathtt{link}}+3t\_{\mathtt{barrier}}+\frac{2}{3}t\_{\mathtt{aggr}}.

Assume that all devices are homogeneous, i.e., t𝚋𝚊𝚛𝚛𝚒𝚎𝚛≈0subscript𝑡𝚋𝚊𝚛𝚛𝚒𝚎𝚛0t\_{\mathtt{barrier}}\approx 0, and t𝚍𝚊𝚝𝚊≈0,t𝚊𝚐𝚐𝚛≈0formulae-sequencesubscript𝑡𝚍𝚊𝚝𝚊0subscript𝑡𝚊𝚐𝚐𝚛0t\_{\mathtt{data}}\approx 0,t\_{\mathtt{aggr}}\approx 0 because the data size is very small. Then we have latencies simplified as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | t𝚜𝚝𝚊𝚛=2​t𝚕𝚒𝚗𝚔<t𝚝𝚛𝚎𝚎=t𝚛𝚒𝚗𝚐=4​t𝚕𝚒𝚗𝚔.subscript𝑡𝚜𝚝𝚊𝚛2subscript𝑡𝚕𝚒𝚗𝚔subscript𝑡𝚝𝚛𝚎𝚎subscript𝑡𝚛𝚒𝚗𝚐4subscript𝑡𝚕𝚒𝚗𝚔t\_{\mathtt{star}}=2t\_{\mathtt{link}}<t\_{\mathtt{tree}}=t\_{\mathtt{ring}}=4t\_{\mathtt{link}}. |  | (11) |

Thus, the star-based allreduce is the most efficient method because it minimizes link latency.

### A.2 A simple-to-use memory scheduler

In our implementation, a context manager is used to ensure that the required block weights are loaded correctly and unload the used weights to free up memory for subsequent blocks. This simplifies the deployment of large-scale LLMs on low-memory edge devices, requiring just one additional line of code:

[⬇](data:text/plain;base64,d2l0aCBtZW1vcnlfbWFuYWdlci53YWl0X2FuZF9yZWxlYXNlKGYic2VsZl9hdHRuLjAiKToKICAgIGhpZGRlbl9zdGF0ZXMgPSBzZWxmX2F0dG4oaGlkZGVuX3N0YXRlcyk=)

1with memory\_manager.wait\_and\_release(f"self\_attn.0"):

2 hidden\_states = self\_attn(hidden\_states)

### A.3 Proof of proposition 3

We start with the first attention block and end with the final FFN block.

Time slot 1 (attention computation): In this initialization step, 𝑾𝚊𝚝𝚝𝚗1superscriptsubscript𝑾𝚊𝚝𝚝𝚗1{\bm{W}}\_{\mathtt{attn}}^{1} must be loaded before computing the first attention block, taking τ𝚊𝚝𝚝𝚗+t𝚊𝚝𝚝𝚗subscript𝜏𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚝𝚝𝚗\tau\_{\mathtt{attn}}+t\_{\mathtt{attn}}. During the computation time t𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚝𝚝𝚗t\_{\mathtt{attn}}, the next FFN weights, 𝑾𝚏𝚏𝚗1superscriptsubscript𝑾𝚏𝚏𝚗1{\bm{W}}\_{\mathtt{ffn}}^{1}, are loading in parallel.

Time slot 2 (allreduce): The attention block is followed by allreduce communication, which takes t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎t\_{\mathtt{all\\_reduce}}, with the next FFN weights, 𝑾𝚏𝚏𝚗1superscriptsubscript𝑾𝚏𝚏𝚗1{\bm{W}}\_{\mathtt{ffn}}^{1}, loading in parallel.

Time slot 3 (FFN computation): By this time, the FFN weights 𝑾𝚏𝚏𝚗1superscriptsubscript𝑾𝚏𝚏𝚗1{\bm{W}}\_{\mathtt{ffn}}^{1} should be fully loaded. If not, the computation must wait for loading to complete. Let t′=t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗superscript𝑡′subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗t^{\prime}=t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}, if t′≥0superscript𝑡′0t^{\prime}\geq 0, no blocking occurs; otherwise, the computation is delayed by |t′|superscript𝑡′|t^{\prime}|. Once loaded, compute the FFN block in t𝚏𝚏𝚗subscript𝑡𝚏𝚏𝚗t\_{\mathtt{ffn}}.

During this time slot, the waiting, computation of the current FFN block and the weight loading of the next attention block occur simultaneously. By the time the current FFN block finishes, the next attention block’s weights 𝑾𝚊𝚝𝚝𝚗2superscriptsubscript𝑾𝚊𝚝𝚝𝚗2{\bm{W}}\_{\mathtt{attn}}^{2} have been loading for max⁡{0,t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗}+t𝚏𝚏𝚗0subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗subscript𝑡𝚏𝚏𝚗\max\{0,t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}\}+t\_{\mathtt{ffn}}.

Time slot 4 (allreduce): The FFN block is followed by allreduce communication, which takes t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎t\_{\mathtt{all\\_reduce}}, with the next attention weights, 𝑾𝚊𝚝𝚝𝚗2superscriptsubscript𝑾𝚊𝚝𝚝𝚗2{\bm{W}}\_{\mathtt{attn}}^{2}, loading in parallel.

Time slot 5 (attention computation): Ensure that the attention weights 𝑾𝚊𝚝𝚝𝚗2superscriptsubscript𝑾𝚊𝚝𝚝𝚗2{\bm{W}}\_{\mathtt{attn}}^{2} are fully loaded. Let t′=max⁡{0,t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗}+t𝚏𝚏𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚊𝚝𝚝𝚗superscript𝑡′0subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗t^{\prime}=\max\{0,t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}\}+t\_{\mathtt{ffn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{attn}}. If t′≥0superscript𝑡′0t^{\prime}\geq 0, the computation proceeds without blocking. Then, 𝑾𝚊𝚝𝚝𝚗2superscriptsubscript𝑾𝚊𝚝𝚝𝚗2{\bm{W}}\_{\mathtt{attn}}^{2} is computed in t𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚝𝚝𝚗t\_{\mathtt{attn}}, and the next FFN weights 𝑾𝚏𝚏𝚗2superscriptsubscript𝑾𝚏𝚏𝚗2{\bm{W}}\_{\mathtt{ffn}}^{2} have been loading for max⁡{0,max⁡{0,t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗}+t𝚏𝚏𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚊𝚝𝚝𝚗}+t𝚊𝚝𝚝𝚗00subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚝𝚝𝚗\max\{0,\max\{0,t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}\}+t\_{\mathtt{ffn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{attn}}\}+t\_{\mathtt{attn}}.

Time slot 6 (allreduce): The allreduce communication takes t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎t\_{\mathtt{all\\_reduce}}, while the next FFN weights 𝑾𝚏𝚏𝚗2superscriptsubscript𝑾𝚏𝚏𝚗2{\bm{W}}\_{\mathtt{ffn}}^{2} are loading in parallel.

Time slot 7 (FFN computation): Ensure that the FFN weights 𝑾𝚏𝚏𝚗2superscriptsubscript𝑾𝚏𝚏𝚗2{\bm{W}}\_{\mathtt{ffn}}^{2} are fully loaded. Let t′=max⁡{0,max⁡{0,t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗}+t𝚏𝚏𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚊𝚝𝚝𝚗}+t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗superscript𝑡′00subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗t^{\prime}=\max\{0,\max\{0,t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}\}+t\_{\mathtt{ffn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{attn}}\}+t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}. If t′≥0superscript𝑡′0t^{\prime}\geq 0, the computation proceeds without blocking.

This process repeats, until the generation task is finished.

For the system to reach a steady state where computation is not blocked by weight loading at any time, the following conditions must hold.

Case 1: t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗≥0subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗0t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}\geq 0.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 3 (l=1𝑙1l=1): | t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗≥0,subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗0\displaystyle t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}\geq 0, |  | (12) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 5 (l=1𝑙1l=1): | t𝚊𝚝𝚝𝚗+t𝚏𝚏𝚗+2​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗−τ𝚊𝚝𝚝𝚗≥0,subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗2subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗subscript𝜏𝚊𝚝𝚝𝚗0\displaystyle t\_{\mathtt{attn}}+t\_{\mathtt{ffn}}+2t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}-\tau\_{\mathtt{attn}}\geq 0, |  | (13) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 7 (l=2𝑙2l=2): | 2​t𝚊𝚝𝚝𝚗+t𝚏𝚏𝚗+3​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−2​τ𝚏𝚏𝚗−τ𝚊𝚝𝚝𝚗≥0,2subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗3subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎2subscript𝜏𝚏𝚏𝚗subscript𝜏𝚊𝚝𝚝𝚗0\displaystyle 2t\_{\mathtt{attn}}+t\_{\mathtt{ffn}}+3t\_{\mathtt{all\\_reduce}}-2\tau\_{\mathtt{ffn}}-\tau\_{\mathtt{attn}}\geq 0, |  | (14) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 9 (l=2𝑙2l=2): | 2​t𝚊𝚝𝚝𝚗+2​t𝚏𝚏𝚗+4​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−2​τ𝚏𝚏𝚗−2​τ𝚊𝚝𝚝𝚗≥0.2subscript𝑡𝚊𝚝𝚝𝚗2subscript𝑡𝚏𝚏𝚗4subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎2subscript𝜏𝚏𝚏𝚗2subscript𝜏𝚊𝚝𝚝𝚗0\displaystyle 2t\_{\mathtt{attn}}+2t\_{\mathtt{ffn}}+4t\_{\mathtt{all\\_reduce}}-2\tau\_{\mathtt{ffn}}-2\tau\_{\mathtt{attn}}\geq 0. |  | (15) |

We repeat these conditions and derive the following patterns.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | t𝚊𝚝𝚝𝚗+t𝚏𝚏𝚗+2​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗2subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle t\_{\mathtt{attn}}+t\_{\mathtt{ffn}}+2t\_{\mathtt{all\\_reduce}} | ≥τ𝚏𝚏𝚗+τ𝚊𝚝𝚝𝚗,absentsubscript𝜏𝚏𝚏𝚗subscript𝜏𝚊𝚝𝚝𝚗\displaystyle\geq\tau\_{\mathtt{ffn}}+\tau\_{\mathtt{attn}}, |  | (16) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+(l−1)⋅t𝚏𝚏𝚗+(2​l−1)⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙1subscript𝑡𝚏𝚏𝚗⋅2𝑙1subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle l\cdot t\_{\mathtt{attn}}+(l-1)\cdot t\_{\mathtt{ffn}}+(2l-1)\cdot t\_{\mathtt{all\\_reduce}} | ≥l⋅τ𝚏𝚏𝚗+(l−1)⋅τ𝚊𝚝𝚝𝚗.absent⋅𝑙subscript𝜏𝚏𝚏𝚗⋅𝑙1subscript𝜏𝚊𝚝𝚝𝚗\displaystyle\geq l\cdot\tau\_{\mathtt{ffn}}+(l-1)\cdot\tau\_{\mathtt{attn}}. |  | (17) |

Case 2: t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗<0subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗0t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}<0.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 3 (l=1𝑙1l=1): | t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗<0,subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗0\displaystyle t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}<0, |  | (18) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 5 (l=1𝑙1l=1): | t𝚏𝚏𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚊𝚝𝚝𝚗≥0,subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗0\displaystyle t\_{\mathtt{ffn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{attn}}\geq 0, |  | (19) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 7 (l=2𝑙2l=2): | t𝚊𝚝𝚝𝚗+t𝚏𝚏𝚗+2​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚊𝚝𝚝𝚗−τ𝚏𝚏𝚗≥0,subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗2subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗subscript𝜏𝚏𝚏𝚗0\displaystyle t\_{\mathtt{attn}}+t\_{\mathtt{ffn}}+2t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{attn}}-\tau\_{\mathtt{ffn}}\geq 0, |  | (20) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 9 (l=2𝑙2l=2): | t𝚊𝚝𝚝𝚗+2​t𝚏𝚏𝚗+3​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−2​τ𝚊𝚝𝚝𝚗−τ𝚏𝚏𝚗≥0,subscript𝑡𝚊𝚝𝚝𝚗2subscript𝑡𝚏𝚏𝚗3subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎2subscript𝜏𝚊𝚝𝚝𝚗subscript𝜏𝚏𝚏𝚗0\displaystyle t\_{\mathtt{attn}}+2t\_{\mathtt{ffn}}+3t\_{\mathtt{all\\_reduce}}-2\tau\_{\mathtt{attn}}-\tau\_{\mathtt{ffn}}\geq 0, |  | (21) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time slot 11 (l=3𝑙3l=3): | 2​t𝚊𝚝𝚝𝚗+2​t𝚏𝚏𝚗+4​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−2​τ𝚊𝚝𝚝𝚗−2​τ𝚏𝚏𝚗≥0.2subscript𝑡𝚊𝚝𝚝𝚗2subscript𝑡𝚏𝚏𝚗4subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎2subscript𝜏𝚊𝚝𝚝𝚗2subscript𝜏𝚏𝚏𝚗0\displaystyle 2t\_{\mathtt{attn}}+2t\_{\mathtt{ffn}}+4t\_{\mathtt{all\\_reduce}}-2\tau\_{\mathtt{attn}}-2\tau\_{\mathtt{ffn}}\geq 0. |  | (22) |

Similarly, repeat these conditions and derive the following patterns.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | t𝚊𝚝𝚝𝚗+t𝚏𝚏𝚗+2​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗2subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle t\_{\mathtt{attn}}+t\_{\mathtt{ffn}}+2t\_{\mathtt{all\\_reduce}} | ≥τ𝚏𝚏𝚗+τ𝚊𝚝𝚝𝚗,absentsubscript𝜏𝚏𝚏𝚗subscript𝜏𝚊𝚝𝚝𝚗\displaystyle\geq\tau\_{\mathtt{ffn}}+\tau\_{\mathtt{attn}}, |  | (23) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (l−1)⋅t𝚊𝚝𝚝𝚗+l⋅t𝚏𝚏𝚗+(2​l−1)⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙1subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙subscript𝑡𝚏𝚏𝚗⋅2𝑙1subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle(l-1)\cdot t\_{\mathtt{attn}}+l\cdot t\_{\mathtt{ffn}}+(2l-1)\cdot t\_{\mathtt{all\\_reduce}} | ≥(l−1)⋅τ𝚏𝚏𝚗+l⋅τ𝚊𝚝𝚝𝚗.absent⋅𝑙1subscript𝜏𝚏𝚏𝚗⋅𝑙subscript𝜏𝚊𝚝𝚝𝚗\displaystyle\geq(l-1)\cdot\tau\_{\mathtt{ffn}}+l\cdot\tau\_{\mathtt{attn}}. |  | (24) |

Thus, the proposition is proved.

### A.4 Proof of proposition 4

Let α=l⋅t𝚊𝚝𝚝𝚗+(l−1)⋅t𝚏𝚏𝚗+(2​l−1)⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−l⋅τ𝚏𝚏𝚗−(l−1)⋅τ𝚊𝚝𝚝𝚗>0𝛼⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙1subscript𝑡𝚏𝚏𝚗⋅2𝑙1subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙subscript𝜏𝚏𝚏𝚗⋅𝑙1subscript𝜏𝚊𝚝𝚝𝚗0\alpha=l\cdot t\_{\mathtt{attn}}+(l-1)\cdot t\_{\mathtt{ffn}}+(2l-1)\cdot t\_{\mathtt{all\\_reduce}}-l\cdot\tau\_{\mathtt{ffn}}-(l-1)\cdot\tau\_{\mathtt{attn}}>0, and we derive the following inequality from inequality ([16](#A1.E16 "In A.3 Proof of proposition 3 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+l⋅t𝚏𝚏𝚗+2​l⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−l⋅τ𝚏𝚏𝚗−l⋅τ𝚊𝚝𝚝𝚗>0.⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙subscript𝑡𝚏𝚏𝚗⋅2𝑙subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙subscript𝜏𝚏𝚏𝚗⋅𝑙subscript𝜏𝚊𝚝𝚝𝚗0l\cdot t\_{\mathtt{attn}}+l\cdot t\_{\mathtt{ffn}}+2l\cdot t\_{\mathtt{all\\_reduce}}-l\cdot\tau\_{\mathtt{ffn}}-l\cdot\tau\_{\mathtt{attn}}>0. |  | (25) |

By substituting α𝛼\alpha into this inequality, we have α+t𝚏𝚏𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚊𝚝𝚝𝚗>0.𝛼subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗0\alpha+t\_{\mathtt{ffn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{attn}}>0.
Let α>0>τ𝚊𝚝𝚝𝚗−t𝚏𝚏𝚗−t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎𝛼0subscript𝜏𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\alpha>0>\tau\_{\mathtt{attn}}-t\_{\mathtt{ffn}}-t\_{\mathtt{all\\_reduce}}, we obtain the first condition:

|  |  |  |  |
| --- | --- | --- | --- |
|  | t𝚏𝚏𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎>τ𝚊𝚝𝚝𝚗.subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗t\_{\mathtt{ffn}}+t\_{\mathtt{all\\_reduce}}>\tau\_{\mathtt{attn}}. |  | (26) |

Let β=t𝚏𝚏𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚊𝚝𝚝𝚗>0𝛽subscript𝑡𝚏𝚏𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚊𝚝𝚝𝚗0\beta=t\_{\mathtt{ffn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{attn}}>0, and substitute β𝛽\beta into inequality ([16](#A1.E16 "In A.3 Proof of proposition 3 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices")), then we have β+t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−τ𝚏𝚏𝚗>0𝛽subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗0\beta+t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-\tau\_{\mathtt{ffn}}>0. Let β>0>τ𝚏𝚏𝚗−t𝚊𝚝𝚝𝚗−t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎𝛽0subscript𝜏𝚏𝚏𝚗subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\beta>0>\tau\_{\mathtt{ffn}}-t\_{\mathtt{attn}}-t\_{\mathtt{all\\_reduce}}, we obtain the second condition:

|  |  |  |  |
| --- | --- | --- | --- |
|  | t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎>τ𝚏𝚏𝚗.subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎subscript𝜏𝚏𝚏𝚗t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}>\tau\_{\mathtt{ffn}}. |  | (27) |

Thus, the proposition is proved.

### A.5 Proof of proposition 5

In this section, we analyze the peak memory footprint on both the master and worker nodes to estimate the largest model size that our memory scheduler can handle.

Let us use the Llama model as an example, assume the vocabulary size is v𝑣v, hidden size is hℎh, number of attention heads is a𝑎a, number of key-value heads is b𝑏b, and intermediate size is s𝑠s. Let 𝒑=[p1,p2,⋯,pn]𝒑

subscript𝑝1subscript𝑝2⋯subscript𝑝𝑛{\bm{p}}=[p\_{1},p\_{2},\cdots,p\_{n}] be a vector representing the proportion of parameters handled by n𝑛n devices, and w𝑤w be the window size of the memory scheduler. Following the block definition in Figure [2](#S3.F2 "Figure 2 ‣ 3.1 The Parallel Framework Design of TPI-LLM System ‣ 3 TPI-LLM Framework with Sliding Window Memory Scheduling ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), the parameter counts for each block are detailed in Table [4](#A1.T4 "Table 4 ‣ A.5 Proof of proposition 5 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"):

Table 4: Parameter counts for the main blocks (e.g., n=4𝑛4n=4, pi=0.25subscript𝑝𝑖0.25p\_{i}=0.25, Llama 2-7B).

|  |  |  |
| --- | --- | --- |
| Block | Parameters | Block Size |
| Preprocess | h​vℎ𝑣hv | 500 MB |
| Attention | 2​(a+b)​h2​pi/a+h2𝑎𝑏superscriptℎ2subscript𝑝𝑖𝑎ℎ2(a+b)h^{2}p\_{i}/a+h | 64 MB |
| FFN | 3​h​s​pi+h3ℎ𝑠subscript𝑝𝑖ℎ3hsp\_{i}+h | 129 MB |
| Postprocess | h​v+hℎ𝑣ℎhv+h | 500 MB |

The memory footprint is affected by parameters, activation storage, temporary tensors, memory management, and caching, making precise quantification challenging. To estimate peak memory, we apply an empirical rule: multiply the parameter size by a scaling factor γ𝛾\gamma.

Figure 8: Illustration of the memory window at the peak memory footprint.

From the memory window at the peak memory footprint shown in Figure [8](#A1.F8 "Figure 8 ‣ A.5 Proof of proposition 5 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), we can derive the following equations.

|  |  |  |
| --- | --- | --- |
|  | M𝚖𝚊𝚜𝚝𝚎𝚛=γ×{h​v+h,if ​w=12​h​v+h,if ​w=22​h​v+h+⌊w−22⌋​(2​(1+ba)​h2​pi+h)+⌊w−12⌋​(3​h​s​pi+h),if ​w≥3subscript𝑀𝚖𝚊𝚜𝚝𝚎𝚛𝛾casesℎ𝑣ℎif 𝑤12ℎ𝑣ℎif 𝑤22ℎ𝑣ℎ𝑤2221𝑏𝑎superscriptℎ2subscript𝑝𝑖ℎ𝑤123ℎ𝑠subscript𝑝𝑖ℎif 𝑤3M\_{\mathtt{master}}=\gamma\times\begin{cases}hv+h,&\text{if }w=1\\ 2hv+h,&\text{if }w=2\\ 2hv+h+\left\lfloor\frac{w-2}{2}\right\rfloor\left(2(1+\frac{b}{a})h^{2}p\_{i}+h\right)+\left\lfloor\frac{w-1}{2}\right\rfloor(3hsp\_{i}+h),&\text{if }w\geq 3\end{cases} |  |

For any worker node, the memory footprint does not include the preprocess and postprocess blocks. Therefore, the peak memory footprint M𝚠𝚘𝚛𝚔𝚎𝚛subscript𝑀𝚠𝚘𝚛𝚔𝚎𝚛M\_{\mathtt{worker}} can be expressed as:

|  |  |  |
| --- | --- | --- |
|  | M𝚠𝚘𝚛𝚔𝚎𝚛=γ×(⌊w2⌋​(2​(1+ba)​h2​pi+h)+⌊w+12⌋​(3​h​s​pi+h)).subscript𝑀𝚠𝚘𝚛𝚔𝚎𝚛𝛾𝑤221𝑏𝑎superscriptℎ2subscript𝑝𝑖ℎ𝑤123ℎ𝑠subscript𝑝𝑖ℎM\_{\mathtt{worker}}=\gamma\times\left(\left\lfloor\frac{w}{2}\right\rfloor(2(1+\frac{b}{a})h^{2}p\_{i}+h)+\left\lfloor\frac{w+1}{2}\right\rfloor(3hsp\_{i}+h)\right). |  |

Thus, the proposition is proved.

### A.6 Proof of proposition 6

When the memory scheduler reaches a steady state, the overlap between computation, communication, and disk I/O is optimized, ensuring that weights are always pre-loaded before they are needed for computations. However, if disk I/O becomes a bottleneck and disrupts the steady state (e.g., due to high disk latency), the scheduler must adapt by selectively retaining certain blocks in memory to reduce disk access frequency.

In our preliminary experiments, we measured t𝚊𝚝𝚝𝚗=11subscript𝑡𝚊𝚝𝚝𝚗11t\_{\mathtt{attn}}=11 ms, t𝚏𝚏𝚗=17subscript𝑡𝚏𝚏𝚗17t\_{\mathtt{ffn}}=17 ms, τ𝚊𝚝𝚝𝚗=18subscript𝜏𝚊𝚝𝚝𝚗18\tau\_{\mathtt{attn}}=18 ms, τ𝚏𝚏𝚗=30subscript𝜏𝚏𝚏𝚗30\tau\_{\mathtt{ffn}}=30 ms, and observed that FFN blocks generally exhibit higher computation and weight loading latency. By retaining some FFN blocks in memory, we can reduce the need to reload large weights.

Let the memory scheduler retain one FFN block in memory every T𝑇T FFN blocks, and

|  |  |  |
| --- | --- | --- |
|  | 𝕀{l=1+k​T}={1,if ​l=1+k​T​ and ​k∈ℤ≥0,0,otherwise.subscript𝕀𝑙1𝑘𝑇cases1if 𝑙1𝑘𝑇 and 𝑘ℤ00otherwise\mathbb{I}\_{\{l=1+kT\}}=\begin{cases}1,&\text{if }l=1+kT\text{ and }k\in\mathbb{Z}{\geq 0},\\ 0,&\text{otherwise}.\end{cases} |  |

Similar to the analysis in Appendix [A.3](#A1.SS3 "A.3 Proof of proposition 3 ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | Time slot 3 (l=1𝑙1l=1): | t𝚊𝚝𝚝𝚗+t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−(1−𝕀{l=1+k​T})​τ𝚏𝚏𝚗≥0,subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎1subscript𝕀𝑙1𝑘𝑇subscript𝜏𝚏𝚏𝚗0\displaystyle t\_{\mathtt{attn}}+t\_{\mathtt{all\\_reduce}}-(1-\mathbb{I}\_{\{l=1+kT\}})\tau\_{\mathtt{ffn}}\geq 0, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Time slot 5 (l=1𝑙1l=1): | t𝚊𝚝𝚝𝚗+t𝚏𝚏𝚗+2​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−(1−𝕀{l=1+k​T})​τ𝚏𝚏𝚗−τ𝚊𝚝𝚝𝚗≥0,subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗2subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎1subscript𝕀𝑙1𝑘𝑇subscript𝜏𝚏𝚏𝚗subscript𝜏𝚊𝚝𝚝𝚗0\displaystyle t\_{\mathtt{attn}}+t\_{\mathtt{ffn}}+2t\_{\mathtt{all\\_reduce}}-(1-\mathbb{I}\_{\{l=1+kT\}})\tau\_{\mathtt{ffn}}-\tau\_{\mathtt{attn}}\geq 0, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Time slot 7 (l=2𝑙2l=2): | 2​t𝚊𝚝𝚝𝚗+t𝚏𝚏𝚗+3​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−∑i=12(1−𝕀{i=1+k​T})​τ𝚏𝚏𝚗−τ𝚊𝚝𝚝𝚗≥0,2subscript𝑡𝚊𝚝𝚝𝚗subscript𝑡𝚏𝚏𝚗3subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎superscriptsubscript𝑖121subscript𝕀𝑖1𝑘𝑇subscript𝜏𝚏𝚏𝚗subscript𝜏𝚊𝚝𝚝𝚗0\displaystyle 2t\_{\mathtt{attn}}+t\_{\mathtt{ffn}}+3t\_{\mathtt{all\\_reduce}}-\sum\_{i=1}^{2}(1-\mathbb{I}\_{\{i=1+kT\}})\tau\_{\mathtt{ffn}}-\tau\_{\mathtt{attn}}\geq 0, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Time slot 9 (l=2𝑙2l=2): | 2​t𝚊𝚝𝚝𝚗+2​t𝚏𝚏𝚗+4​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−∑i=12(1−𝕀{i=1+k​T})​τ𝚏𝚏𝚗−2​τ𝚊𝚝𝚝𝚗≥0,2subscript𝑡𝚊𝚝𝚝𝚗2subscript𝑡𝚏𝚏𝚗4subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎superscriptsubscript𝑖121subscript𝕀𝑖1𝑘𝑇subscript𝜏𝚏𝚏𝚗2subscript𝜏𝚊𝚝𝚝𝚗0\displaystyle 2t\_{\mathtt{attn}}+2t\_{\mathtt{ffn}}+4t\_{\mathtt{all\\_reduce}}-\sum\_{i=1}^{2}(1-\mathbb{I}\_{\{i=1+kT\}})\tau\_{\mathtt{ffn}}-2\tau\_{\mathtt{attn}}\geq 0, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Time slot 11 (l=3𝑙3l=3): | 3​t𝚊𝚝𝚝𝚗+2​t𝚏𝚏𝚗+5​t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−∑i=13(1−𝕀{i=1+k​T})​τ𝚏𝚏𝚗−2​τ𝚊𝚝𝚝𝚗≥0.3subscript𝑡𝚊𝚝𝚝𝚗2subscript𝑡𝚏𝚏𝚗5subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎superscriptsubscript𝑖131subscript𝕀𝑖1𝑘𝑇subscript𝜏𝚏𝚏𝚗2subscript𝜏𝚊𝚝𝚝𝚗0\displaystyle 3t\_{\mathtt{attn}}+2t\_{\mathtt{ffn}}+5t\_{\mathtt{all\\_reduce}}-\sum\_{i=1}^{3}(1-\mathbb{I}\_{\{i=1+kT\}})\tau\_{\mathtt{ffn}}-2\tau\_{\mathtt{attn}}\geq 0. |  |

By repeating these conditions, we derive the following patterns:

|  |  |  |  |
| --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+l⋅t𝚏𝚏𝚗+2​l⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−∑i=1l(1−𝕀{i=1+k​T})​τ𝚏𝚏𝚗−l⋅τ𝚊𝚝𝚝𝚗⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙subscript𝑡𝚏𝚏𝚗⋅2𝑙subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎superscriptsubscript𝑖1𝑙1subscript𝕀𝑖1𝑘𝑇subscript𝜏𝚏𝚏𝚗⋅𝑙subscript𝜏𝚊𝚝𝚝𝚗\displaystyle l\cdot t\_{\mathtt{attn}}+l\cdot t\_{\mathtt{ffn}}+2l\cdot t\_{\mathtt{all\\_reduce}}-\sum\_{i=1}^{l}(1-\mathbb{I}\_{\{i=1+kT\}})\tau\_{\mathtt{ffn}}-l\cdot\tau\_{\mathtt{attn}} | ≥0,absent0\displaystyle\geq 0, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+(l−1)⋅t𝚏𝚏𝚗+(2​l−1)⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎−∑i=1l(1−𝕀{i=1+k​T})​τ𝚏𝚏𝚗−(l−1)⋅τ𝚊𝚝𝚝𝚗⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙1subscript𝑡𝚏𝚏𝚗⋅2𝑙1subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎superscriptsubscript𝑖1𝑙1subscript𝕀𝑖1𝑘𝑇subscript𝜏𝚏𝚏𝚗⋅𝑙1subscript𝜏𝚊𝚝𝚝𝚗\displaystyle l\cdot t\_{\mathtt{attn}}+(l-1)\cdot t\_{\mathtt{ffn}}+(2l-1)\cdot t\_{\mathtt{all\\_reduce}}-\sum\_{i=1}^{l}(1-\mathbb{I}\_{\{i=1+kT\}})\tau\_{\mathtt{ffn}}-(l-1)\cdot\tau\_{\mathtt{attn}} | ≥0.absent0\displaystyle\geq 0. |  |

Since ∑i=1l𝕀{i=1+k​T}=⌈lT⌉superscriptsubscript𝑖1𝑙subscript𝕀𝑖1𝑘𝑇𝑙𝑇\sum\_{i=1}^{l}\mathbb{I}\_{\{i=1+kT\}}=\left\lceil\frac{l}{T}\right\rceil, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+l⋅t𝚏𝚏𝚗+2​l⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙subscript𝑡𝚏𝚏𝚗⋅2𝑙subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle l\cdot t\_{\mathtt{attn}}+l\cdot t\_{\mathtt{ffn}}+2l\cdot t\_{\mathtt{all\\_reduce}} | ≥(l−⌈lT⌉)⋅τ𝚏𝚏𝚗+l⋅τ𝚊𝚝𝚝𝚗,absent⋅𝑙𝑙𝑇subscript𝜏𝚏𝚏𝚗⋅𝑙subscript𝜏𝚊𝚝𝚝𝚗\displaystyle\geq(l-\left\lceil\frac{l}{T}\right\rceil)\cdot\tau\_{\mathtt{ffn}}+l\cdot\tau\_{\mathtt{attn}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | l⋅t𝚊𝚝𝚝𝚗+(l−1)⋅t𝚏𝚏𝚗+(2​l−1)⋅t𝚊𝚕𝚕​\_​𝚛𝚎𝚍𝚞𝚌𝚎⋅𝑙subscript𝑡𝚊𝚝𝚝𝚗⋅𝑙1subscript𝑡𝚏𝚏𝚗⋅2𝑙1subscript𝑡𝚊𝚕𝚕\_𝚛𝚎𝚍𝚞𝚌𝚎\displaystyle l\cdot t\_{\mathtt{attn}}+(l-1)\cdot t\_{\mathtt{ffn}}+(2l-1)\cdot t\_{\mathtt{all\\_reduce}} | ≥(l−⌈lT⌉)⋅τ𝚏𝚏𝚗+(l−1)⋅τ𝚊𝚝𝚝𝚗.absent⋅𝑙𝑙𝑇subscript𝜏𝚏𝚏𝚗⋅𝑙1subscript𝜏𝚊𝚝𝚝𝚗\displaystyle\geq(l-\left\lceil\frac{l}{T}\right\rceil)\cdot\tau\_{\mathtt{ffn}}+(l-1)\cdot\tau\_{\mathtt{attn}}. |  |

Thus, the proposition is proved.

### A.7 Klonet testbed

One of our testbed, as shown in Figure [9](#A1.F9 "Figure 9 ‣ A.7 Klonet testbed ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), was built upon Klonet (Ma et al., [2024](#bib.bib14)) to create an edge network environment. Klonet is a network emulation platform designed to support the development and testing of new network protocols and applications in a realistic environment. It can emulate various network scenarios, such as wireless, mobile, satellite, and optical networks, and provide fine-grained control over the network parameters, such as bandwidth, delay, jitter, and packet loss. It can also integrate with real devices and applications, such as routers, switches, sensors, and smartphones, to create hybrid network experiments. Klonet is based on the Linux operating system and uses virtualization and containerization technologies to create isolated network nodes and links. It provides both GUI and CLI to help users configure and manage their network experiments.

Figure 9: Testbed built upon Klonet.

This testbed includes 8 user devices (devices 1 to 8), 8 home gateways (routers 1 to 8), and 1 core router (router 9). User devices connect to their home gateways via wired or wireless connections, and these home gateways are interconnected through routers or switches in the edge network. This topology reflects real-world household network interconnections. In addition, the CPU cores, memory, swap limits, bandwidth, and latency settings in Section [4](#S4 "4 Experiments ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") are based on measurements from the authors’ edge network.

### A.8 Configurations of the used models

Table 5: Configurations of the used Llama models.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model (FP32) | Layers | Params | Hidden Size | Heads | KV Heads | Required Mem |
| Llama 2-3B | 26 | 3 billion | 3200 | 32 | – | 14 GB |
| Llama 2-7B | 32 | 7 billion | 4096 | 32 | – | 26 GB |
| Llama 2-13B | 40 | 13 billion | 5120 | 40 | – | 50 GB |
| Llama 2-70B | 80 | 70 billion | 8192 | 64 | 8 | 257 GB |
| Llama 3.1-8B | 32 | 8 billion | 4096 | 32 | 8 | 31 GB |
| Llama 3.1-70B | 80 | 70 billion | 8192 | 64 | 8 | 266 GB |
| Yi-34B | 60 | 34 billion | 7168 | 56 | 8 | 130 GB |

### A.9 Peak memory footprint with memory window size 4

Table 6: Peak memory footprint per device with the memory window size set to 4.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Memory Scheduler Disabled (GB) | | | | Memory Scheduler Enabled (GB) | | | |
| Model (FP32) | N=2𝑁2N=2 | N=4𝑁4N=4 | N=6𝑁6N=6 | N=8𝑁8N=8 | N=2𝑁2N=2 | N=4𝑁4N=4 | N=6𝑁6N=6 | N=8𝑁8N=8 |
| Llama 2-3B | 7.3 | 4.3 | 3.2 | 2.8 | 1.7 | 1.5 | 1.5 | 1.5 |
| Llama 2-7B | 13.7 | 7.7 | 5.5 | 4.5 | 2.4 | 2.1 | 1.8 | 1.8 |
| Llama 2-13B | 25.8 | 13.9 | 9.7 | 8.0 | 2.8 | 2.5 | 2.3 | 2.2 |
| Llama 2-70B | 129.9 | 66.5 | 46.7 | 35.0 | 4.5 | 3.1 | 3.1 | 3.1 |
| Llama 3.1-8B | 18.4 | 11.8 | 9.4 | 8.5 | 6.3 | 5.8 | 5.6 | 5.5 |
| Llama 3.1-70B | 137.8 | 74.0 | 51.4 | 42.5 | 10.8 | 10.5 | 11.5 | 11.4 |
| Yi-34B | 67 | 36.4 | 23.9 | 20.4 | 6.0 | 5.6 | 5.3 | 5.2 |

### A.10 Real testbed and configurations

The real testbed consists of 4 laptops, all connected via a local Wi-Fi router, as shown in Figure [10](#A1.F10 "Figure 10 ‣ A.10 Real testbed and configurations ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"). Table [7](#A1.T7 "Table 7 ‣ A.10 Real testbed and configurations ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices") details the hardware and network configurations of these laptops. In this case study, the laptop in the lower right serves as the master, while the other three laptops act as workers. The workers are connected to the master, and the master is currently generating the output sequence. The generated sequence is identical to that of single-server inference.

Figure 10: A real testbed composed of 4 laptops connected via local Wi-Fi.




Table 7: Hardware and network configurations of the laptops.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Device | CPU Model | Cores | Memory | Bandwidth | Latency | CUDA | Number |
| Mac Pro | Apple M1 | 8 | 8 GB | 510 Mbps | 5 ms | No | 1 |
| Mac Air | Intel Core i5 | 4 | 8 GB | 320 Mbps | 7 ms | No | 1 |
| Dell | Intel i7-1165G7 | 8 | 16 GB | 610 Mbps | 3 ms | No | 2 |

### A.11 Case study with 3 laptops

In this case, only 3 out of the 4 laptops are used: one MacBook Pro, one MacBook Air, and one Dell laptop. The results are given in Table [8](#A1.T8 "Table 8 ‣ A.11 Case study with 3 laptops ‣ Appendix A Appendix ‣ TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices"), indicating a slightly higher latency due to reduced parallelism.

Table 8: Comparison of Transformers, Accelerate, Transformers+MS, and TPI-LLM on 3 laptops.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model (FP32) | Transformers | | Accelerate | | Transformers + MS | | TPI-LLM | |
| TTFT  (s) | Latency  (s/token) | TTFT  (s) | Latency  (s/token) | TTFT  (s) | Latency  (s/token) | TTFT  (s) | Latency  (s/token) |
| Llama 2-3B | 61 | 30 | 24 | 16 | 4 | 3 | 3 | 2 |
| Llama 2-7B | 115 | 56 | 30 | 26 | 13 | 8 | 7 | 6 |
| Llama 2-13B | OOM | OOM | OOM | OOM | 22 | 18 | 14 | 12 |
| Llama 3.1-8B | 133 | 65 | 37 | 31 | 20 | 12 | 13 | 9 |
| Yi-34B | OOM | OOM | OOM | OOM | 185 | 55 | 48 | 41 |
