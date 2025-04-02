# Efficiently Serving Large Language Models (LLMs)

## Course Overview

The **Efficiently Serving LLMs** course, offered by [DeepLearning.AI](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/), provides a **practical introduction to optimizing LLM inference**. The course covers key techniques to improve **text generation speed, optimize resource usage, and manage multiple fine-tuned models effectively**.

### **What You'll Learn**
- How LLMs generate text using **auto-regressive token prediction**.
- Key optimization techniques such as **KV caching, continuous batching, and model quantization**.
- Trade-offs in LLM inference to balance **speed and scalability** for multiple users.
- **Low-Rank Adapters (LoRA) and LoRAX framework** for efficient multi-model serving.
- Real-world inference optimizations using Predibase’s **LoRAX**.

**Instructor**: Travis Addair, CTO at Predibase.


## Course Content

### [**1. Text Generation**](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L1-Text_Generation.ipynb)
- **Encoder vs Decoder LLMs** and autoregressive text generation.
- **KV Cache Prefill & Decode**: Using cached key-value tensors to optimize token generation.
- **Notebook Implementation**:
  - Load a pre-trained **GPT-2** model from Hugging Face.
  - Implement **token generation**, **top-k sampling**, and **performance measurement**.
  - Optimize using **KV-caching** and compare execution times.

### [**2. Batching for Efficient Inference**](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L2-Batching.ipynb)
- **Single vs. Multi-Request Generation** and their trade-offs.
- **Throughput vs. Latency** analysis with and without batching.
- **Notebook Implementation**:
  - Implement **batch token generation** with padding and position IDs.
  - Compare performance across different batch sizes.

### [**3. Continuous Batching**](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L3-Continuous_Batching.ipynb)
- Improves efficiency by dynamically swapping completed requests with new ones.
- **Notebook Implementation**:
  - Implements **request queue processing** for continuous batching.
  - Uses `generate_next_token()`, `merge_batches()`, and `filter_batch()` for dynamic management.
  - Measures performance differences compared to traditional batching.

### [**4. Quantization**](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L4-Quantization.ipynb)
- Reduces model memory usage while maintaining accuracy.
- **Notebook Implementation**:
  - Apply **zero-point quantization** to GPT-2 model parameters.
  - Implement functions for **quantization and dequantization**.
  - Compare **memory footprints** before and after quantization.

### [**5. Low-Rank Adaptation (LoRA)**](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L5-Low-Rank_Adaptation.ipynb)
- Efficient fine-tuning technique that reduces the number of trainable parameters.
- **Notebook Implementation**:
  - Define **LoRA matrices (A & B)** and integrate them into a model.
  - Implement **LoraLayer** in PyTorch to modify linear layers.
  - Compare results with and without LoRA applied.

### [**6. Multi-LoRA Inference**](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L6-Multi-LoRA.ipynb)
- Serving multiple LoRA models efficiently in a single deployment.
- **Notebook Implementation**:
  - Implement **loop-based vs. vectorized LoRA computation**.
  - Optimize with `torch.index_select()` for efficient inference.
  - Compare performance benchmarks for different LoRA implementations.

### [**7. LoRAX: Efficient LLM Serving**](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/predibase_lorax.ipynb)
- Framework that combines **continuous batching, quantization, and multi-LoRA inference**.
- **Notebook Implementation**:
  - Implement LoRAX techniques to scale LLM deployments.
  - Compare inference speed across different settings.


## Notebooks
The course includes Jupyter Notebooks that provide hands-on implementations of all optimization techniques:
- [`L1-Text_Generation.ipynb`](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L1-Text_Generation.ipynb) – Covers text generation fundamentals.
- [`L2-Batching.ipynb`](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L2-Batching.ipynb) – Implements efficient batching strategies.
- [`L3-Continuous_Batching.ipynb`](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L3-Continuous_Batching.ipynb) – Demonstrates dynamic request processing.
- [`L4-Quantization.ipynb`](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L4-Quantization.ipynb) – Applies quantization to reduce memory usage.
- [`L5-Low-Rand_Adaptation.ipynb`](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L5-Low-Rank_Adaptation.ipynb) – Implements Low-Rank Adaptation for fine-tuning.
- [`L6-Multi-Lora.ipynb`](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/L6-Multi-LoRA.ipynb) – Optimizes LoRA for multi-model inference.
- [`prediabse_lorax.ipynb`](https://github.com/michaWorku/Efficiently-Serving-LLMs/blob/main/predibase_lorax.ipynb) – Integrates all optimizations into a scalable LLM serving framework.


## Getting Started

### **Prerequisites**
- Python (>=3.7)
- PyTorch (>=1.9)
- Transformers (Hugging Face)
- Matplotlib & NumPy for performance visualization

### **Installation**
```bash
# Clone the repository
git clone https://github.com/your-repo/efficient-llm-serving.git
cd efficient-llm-serving

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Running Notebooks**
```bash
jupyter notebook
```
Open the desired `.ipynb` file and run the cells.


## Resources & References
- [Course Homepage](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)
- [Predibase](https://predibase.com/) – Company behind LoRAX and efficient LLM techniques.
- [Hugging Face](https://huggingface.co/) – Source for pre-trained language models.
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685) – Original paper on Low-Rank Adaptation.
- [Efficient LLM Inference Blog](https://huggingface.co/blog/llm-inference) – Best practices for scaling LLM inference.

