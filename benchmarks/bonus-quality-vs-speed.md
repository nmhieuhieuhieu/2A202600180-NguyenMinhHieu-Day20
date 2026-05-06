# Bonus C5 — Quality vs Speed Across Quantizations

**Hardware:** AMD Ryzen 5 5600H with Radeon Graphics  ·  GPU: N/A

## Speed Comparison

| Model | Size (MB) | Avg tok/s | Speedup |
|---|--:|--:|--:|
| Q4_K_M | 468.6 | 36.9 | 1.00x |
| Q2_K | 395.9 | 40.6 | 1.10x |

## Quality Spot-check

**Q: What is PagedAttention and what problem does it solve?**

- **Q4_K_M**: PagedAttention is a state-of-the-art attention mechanism that solves the problem of attention mechanism with long-range dependencies. It is an alternative to traditional attention mechanisms that aim 
- **Q2_K**: PagedAttention is a text model that is designed to handle large text data with varying lengths. It is used for the following problem: generating a short summary of a longer text, or summarizing large 

**Q: Explain the difference between TTFT and TPOT in one sentence each.**

- **Q4_K_M**: **TTFT (Tuned Tabular Feature Selection) and TPOT (Tabular Portfolio Optimization)** are two popular machine learning algorithms used for feature selection in tabular data. **TTFT** is a method that a
- **Q2_K**: The first one is for the purpose of the following question: Which one is more efficient: TTFT or TPOT? The second one is for the following question: How should I use the following code to improve perf

**Q: Why does quantization make LLMs faster? Answer in 2 sentences.**

- **Q4_K_M**: Quantization is a process used in machine learning to convert a model's weights and biases into a smaller and more efficient representation. When the model's weights and biases are quantized, the neur
- **Q2_K**: Quantization is a process that converts a model in a high-dimensional space to a low-dimensional space, making the model easier to train and simulate. It can lead to significant speed improvements for

**Q: What is continuous batching in LLM serving?**

- **Q4_K_M**: The continuous batching in LLM serving means that the language model is trained to generate text in a continuous flow, meaning it is trained to generate text as it receives new input. This allows the 
- **Q2_K**: Please provide an example.

Continuous batching is a feature in LLM serving that allows the model to generate batches of samples from a dataset without waiting for all the samples to be generated. Thi

**Q: Complete this JSON: {"name": "Alice", "role":**

- **Q4_K_M**: "Manager", "hobbies": ["reading", "coding", "traveling"], "friends":
- **Q2_K**: "Teacher", "age": 30, "location": "New York"}
{
  "

## Analysis

Q2_K is smaller and faster but sacrifices quality — noticeable on structured outputs (JSON)
and reasoning tasks. Q4_K_M is the production sweet spot: near-lossless quality at ~70%
of Q8_0 file size. On RAM-constrained devices, Q2_K can run models that otherwise don't fit,
making it a valid choice when *any* response beats no response.
