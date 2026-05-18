---
arxiv: '2506.13585'
authors:
- MiniMax
- Aili Chen
- Aonian Li
- Bangwei Gong
- Binyang Jiang
- Bo Fei
- Bo Yang
- Boji Shan
- Changqing Yu
- Chao Wang
- Cheng Zhu
- Chengjun Xiao
- Chengyu Du
- Chi Zhang
- Chu Qiao
- Chunhao Zhang
- Chunhui Du
- Congchao Guo
- Da Chen
- Deming Ding
- Dianjun Sun
- Dong Li
- Enwei Jiao
- Haigang Zhou
- Haimo Zhang
- Han Ding
- Haohai Sun
- Haoyu Feng
- Huaiguang Cai
- Haichao Zhu
- Jian Sun
- Jiaqi Zhuang
- Jiaren Cai
- Jiayuan Song
- Jin Zhu
- Jingyang Li
- Jinhao Tian
- Jinli Liu
- Junhao Xu
- Junjie Yan
- Junteng Liu
- Junxian He
- Kaiyi Feng
- Ke Yang
- Kecheng Xiao
- Le Han
- Leyang Wang
- Lianfei Yu
- Liheng Feng
- Lin Li
- Lin Zheng
- Linge Du
- Lingyu Yang
- Lunbin Zeng
- Minghui Yu
- Mingliang Tao
- Mingyuan Chi
- Mozhi Zhang
- Mujie Lin
- Nan Hu
- Nongyu Di
- Peng Gao
- Pengfei Li
- Pengyu Zhao
- Qibing Ren
- Qidi Xu
- Qile Li
- Qin Wang
- Rong Tian
- Ruitao Leng
- Shaoxiang Chen
- Shaoyu Chen
- Shengmin Shi
- Shitong Weng
- Shuchang Guan
- Shuqi Yu
- Sichen Li
- Songquan Zhu
- Tengfei Li
- Tianchi Cai
- Tianrun Liang
- Weiyu Cheng
- Weize Kong
- Wenkai Li
- Xiancai Chen
- Xiangjun Song
- Xiao Luo
- Xiao Su
- Xiaobo Li
- Xiaodong Han
- Xinzhu Hou
- Xuan Lu
- Xun Zou
- Xuyang Shen
- Yan Gong
- Yan Ma
- Yang Wang
- Yiqi Shi
- Yiran Zhong
- Yonghong Duan
- Yongxiang Fu
- Yongyi Hu
- Yu Gao
- Yuanxiang Fan
- Yufeng Yang
- Yuhao Li
- Yulin Hu
- Yunan Huang
- Yunji Li
- Yunzhi Xu
- Yuxin Mao
- Yuxuan Shi
- Yuze Wenren
- Zehan Li
- Zelin Li
- Zhanxu Tian
- Zhengmao Zhu
- Zhenhua Fan
- Zhenzhen Wu
- Zhichao Xu
- Zhihang Yu
- Zhiheng Lyu
- Zhuo Jiang
- Zibo Gao
- Zijia Wu
- Zijian Song
- Zijun Sun
created: '2026-05-18'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2506.13585
  raw: '[[raw/papers/md/2025-minimax-m1-scaling-test-time-compute-efficiently-with-lightning-attention]]'
  source: https://arxiv.org/abs/2506.13585
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-minimax-m1-scaling-test-time-compute-efficiently-with-lightning-attention.md
raw_pdf: raw/papers/pdf/2025-minimax-m1-scaling-test-time-compute-efficiently-with-lightning-attention.pdf
read: false
slug: minimax-m1-scaling-test-time-compute-efficiently-with-lightning-attention
tags:
- type/paper
- status/stub
- mixture-of-experts
- attention
- reinforcement-learning
- long-context
- llm
title: 'MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention'
type: note
updated: '2026-05-18'
year: 2025
---

# MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention

> *MiniMax — Aili Chen, Aonian Li, Bangwei Gong, et al.* — arXiv 2025

## TL;DR

MiniMax-M1 is the first open-weight large-scale **hybrid-attention reasoning model**: a hybrid MoE architecture combined with a *lightning attention* mechanism, built on MiniMax-Text-01 (456B total params, 45.9B active per token). It natively supports a **1M-token context** (8× DeepSeek R1), and lightning attention makes test-time-compute scaling efficient. Trained with large-scale RL on diverse problems including sandboxed real-world software-engineering environments. The paper introduces **CISPO** — an RL algorithm that clips *importance-sampling weights* rather than token updates, outperforming other RL variants; combined with hybrid attention it let full RL training finish on 512 H800 GPUs in three weeks at ~$534,700. Two releases (40K / 80K thinking budgets); comparable or superior to DeepSeek-R1 and Qwen3-235B, with strengths in SWE, tool use, and long context. (Inbox flagged this entry for **CISPO**, the RL algorithm.) (Summary from abstract; note unread.)

## Abstract

We introduce MiniMax-M1, the world's first open-weight, large-scale hybrid-attention reasoning model. MiniMax-M1 is powered by a hybrid Mixture-of-Experts (MoE) architecture combined with a lightning attention mechanism. The model is developed based on our previous MiniMax-Text-01 model, which contains a total of 456 billion parameters with 45.9 billion parameters activated per token. The M1 model natively supports a context length of 1 million tokens, 8x the context size of DeepSeek R1. Furthermore, the lightning attention mechanism in MiniMax-M1 enables efficient scaling of test-time compute. These properties make M1 particularly suitable for complex tasks that require processing long inputs and thinking extensively. MiniMax-M1 is trained using large-scale reinforcement learning (RL) on diverse problems including sandbox-based, real-world software engineering environments. In addition to M1's inherent efficiency advantage for RL training, we propose CISPO, a novel RL algorithm to further enhance RL efficiency. CISPO clips importance sampling weights rather than token updates, outperforming other competitive RL variants. Combining hybrid-attention and CISPO enables MiniMax-M1's full RL training on 512 H800 GPUs to complete in only three weeks, with a rental cost of just $534,700. We release two versions of MiniMax-M1 models with 40K and 80K thinking budgets respectively, where the 40K model represents an intermediate phase of the 80K training. Experiments on standard benchmarks show that our models are comparable or superior to strong open-weight models such as the original DeepSeek-R1 and Qwen3-235B, with particular strengths in complex software engineering, tool utilization, and long-context tasks. We publicly release MiniMax-M1 at https://github.com/MiniMax-AI/MiniMax-M1.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2506.13585>
- PDF: [[raw/papers/pdf/2025-minimax-m1-scaling-test-time-compute-efficiently-with-lightning-attention.pdf]]
- Raw markdown: [[raw/papers/md/2025-minimax-m1-scaling-test-time-compute-efficiently-with-lightning-attention]]
