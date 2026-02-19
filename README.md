# Conversation Summariser — Instruction Fine-Tuning with QLoRA

## Project Overview

This project demonstrates **parameter-efficient fine-tuning** of [Phi-2](https://huggingface.co/microsoft/phi-2) (2.7B parameters) using **QLoRA** for the task of abstractive conversation summarisation. Two distinct training loss strategies were designed, implemented, and benchmarked — showcasing both standard and custom deep learning engineering techniques.

---

## Tech Stack

| Component | Details |
|---|---|
| Language | Python |
| Model Hub | HuggingFace Transformers & Datasets |
| Training Framework | PyTorch |
| Fine-Tuning Method | QLoRA (via PEFT) |
| Training Platform | Kaggle |

---

## Methodology

**Base Model:** Microsoft Phi-2 (2.7B parameters)  
**Fine-Tuning Approach:** QLoRA (Quantized Low-Rank Adaptation) — enabling efficient training on consumer-grade hardware  
**Task:** Abstractive multi-turn conversation summarisation

---

## Training Strategies

### Strategy 1 — Standard Instruction Fine-Tuning

Next-token prediction loss applied across the full sequence: `[Instruction + Conversation + Summary]`.

### Strategy 2 — Custom Loss (Summary-Tokens Only)

Loss computed **exclusively on summary tokens**, masking the instruction and conversation context from gradient updates. This required engineering a custom collate function, a custom training loop, and additional preprocessing columns in the HuggingFace `DatasetDict`.

> This targeted approach focuses the model's learning signal on what matters most — generating high-quality summaries — rather than re-learning the input context.

---

## Prompt Format Ablation

Two instruction formats were benchmarked **before fine-tuning** to identify the stronger baseline for all subsequent experiments.

**Format 1 — Custom Instruction**
```
Instruction: Summarise the following conversation
Conversation: {conversation}
Summary:
```

**Format 2 — Alpaca Style**
```
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruct: Summarise the following Conversation.

{dialogue}

### Output:
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Dataset | DialogueSum (2,000 training examples) |
| Epochs | 2 |
| Batch Size | 1 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 4 |
| QLoRA Rank | 32 |
| QLoRA Dropout | 0.05 |
| Gradient Checkpointing | Enabled |

---

## Results

### Baseline Performance (Pre-Fine-Tuning)

| Metric | Custom Instruction | Alpaca Style |
|---|---|---|
| ROUGE-1 | 0.2157 | **0.2397** |
| ROUGE-2 | 0.0698 | 0.0606 |
| ROUGE-L | 0.1659 | **0.2052** |
| ROUGE-Lsum | 0.1740 | **0.2157** |
| BLEU | 0.0388 | **0.0489** |

The Alpaca-style format demonstrated consistent superiority across the primary metrics and was selected as the input template for all fine-tuning experiments.

---

### Post-Fine-Tuning Performance

| Metric | Baseline (Alpaca) | Strategy 1 — Standard Loss | Strategy 2 — Custom Loss | Improvement (S2 vs Baseline) |
|---|---|---|---|---|
| ROUGE-1 | 0.2397 | 0.4239 | **0.4335** | **+80%** |
| ROUGE-2 | 0.0606 | 0.2002 | **0.2095** | **+245%** |
| ROUGE-L | 0.2052 | 0.3675 | **0.3770** | **+84%** |
| ROUGE-Lsum | 0.2157 | 0.3682 | **0.3770** | **+75%** |
| BLEU | 0.0489 | 0.1739 | **0.1793** | **+266%** |

Strategy 2 (Custom Loss) outperforms the standard fine-tuning approach across all metrics, achieving an **~80% improvement on ROUGE-1** over the pre-fine-tuning baseline. The additional engineering investment in the custom training pipeline is clearly validated by the results.

---

## Engineering Highlights

- Designed and implemented a **custom PyTorch collate function** to mask non-summary tokens during loss computation
- Built a **custom training loop** to support token-level loss masking, going beyond the standard HuggingFace `Trainer` API
- Conducted a **systematic prompt format ablation study** prior to fine-tuning to establish the strongest baseline
- Evaluated model quality using both **ROUGE** and **BLEU** metrics for comprehensive, multi-faceted assessment

---

## Qualitative Comparison

The table below illustrates model outputs across all three conditions for representative conversations from the test set.

| Conversation | Without Fine-Tuning | Strategy 1: Standard Fine-Tuning | Strategy 2: Custom Loss Fine-Tuning |
|---|---|---|---|
| #Person1#: Hello, how are you doing today? #Person2#: I've been having trouble breathing lately. #Person1#: Have you had any type of cold lately? #Person2#: No, I haven't had a cold. I just have a heavy feeling in my chest when I try to breathe. #Person1#: Do you have any allergies that you know of? #Person2#: No, I don't have any allergies that I know of. #Person1#: Does this happen all the time or mostly when you are active? #Person2#: It happens a lot when I work out. #Person1#: I am going to send you to a pulmonary specialist who can run tests on you for asthma. #Person2#: Thank you for your help, doctor. | #Person1#: Hello, how are you doing today? #Person2#: I | #Person2# has trouble breathing and #Person1# asks #Person2# some questions. | #Person2# has trouble breathing and #Person1# recommends #Person2# to see a |
| #Person1#: Hey Jimmy. Let's go workout later today. #Person2#: Sure. What time do you want to go? #Person1#: How about at 3:30? #Person2#: That sounds good. Today we work on Legs and forearm. #Person1#: Hey. I just played basketball earlier, so my legs are a little sore. Let's work out on arms and stomach today. #Person2#: I'm on a weekly schedule. You're messing everything up. #Person1#: C'mon. We're only switching two days. You can do legs on Friday. #Person2#: Aright. I'll meet you at the gym at 3:30 then. | #Person1#: Hey Jimmy. Let's go workout later today. #Person2# | #Person1# and Jimmy decide to workout together. #Person1# suggests working out on arms | #Person1# and Jimmy decide to workout together. #Person1#'s legs are sore |
| #Person1#: I need to stop eating such unhealthy foods. #Person2#: I know what you mean. I've started eating better myself. #Person1#: What foods do you eat now? #Person2#: I tend to stick to fruits, vegetables, and chicken. #Person1#: Those are the only things you eat? #Person2#: That's basically what I eat. #Person1#: Why aren't you eating anything else? #Person2#: Well, fruits and vegetables are very healthy. #Person1#: And the chicken? #Person2#: It's really healthy to eat when you bake it. #Person1#: I guess that does sound a lot healthier. | #Person1#: I need to stop eating such unhealthy foods. #Person2#: | #Person1# and #Person2# talk about eating healthy foods. #Person2# tells | #Person1# and #Person2# talk about their eating habits. #Person2# eats |
| #Person1#: Do you believe in UFOs? #Person2#: Of course, they are out there. #Person1#: But I never saw them. #Person2#: Are you stupid? They are called UFOs, so not everybody can see them. #Person1#: You mean that you can see them. #Person2#: That's right. I can see them in my dreams. #Person1#: They come to the earth? #Person2#: No. Their task is to send the aliens here from the outer space. #Person1#: Aliens from the outer space? Do you talk to them? What do they look like? #Person2#: OK, OK, one by one, please! They look like robots, but they can speak. Their mission is to make friends with human beings. #Person1#: That means that you talk to them? In which language? #Person2#: Of course in English, they learn English on Mars too. #Person1#: Wow. Sounds fantastic! | #Person1#: Do you believe in UFOs? #Person2#: Of course | #Person2# believes in UFOs and can see them in dreams. #Person2# tells # | #Person1#: Wow. Sounds fantastic! ### Output: #Person2# believes in UFOs and can see them in dreams. #Person2# tells # |

---

## Model Weights & Credits

Fine-tuned model weights are publicly available on HuggingFace. Both models were trained on the [`neil-code/dialogsum-test`](https://huggingface.co/datasets/neil-code/dialogsum-test) dataset using Kaggle's GPU environment.

| Model | HuggingFace Link |
|---|---|
| Strategy 1 — Standard Loss | [Pankaj121212/phi-2-nex](https://huggingface.co/Pankaj121212/phi-2-nex) |
| Strategy 2 — Custom Loss | [Pankaj121212/phi-2-cus](https://huggingface.co/Pankaj121212/phi-2-cus) |
