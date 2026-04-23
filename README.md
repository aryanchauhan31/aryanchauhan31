# 👋 Hi, I'm Aryan 

I'm an AI Engineer at Charlemagne Labs and a recent MS in Computer Engineering graduate from NYU, passionate about building efficient and scalable AI systems. My work focuses on LLM inference optimization (vLLM, quantization), GPU kernel programming (CUDA C++), RAG architectures, and multimodal models. An active open-source contributor—most recently to the 🤗 transformers library—I am currently seeking full-time Machine Learning Engineering roles to build high-performance AI infrastructure.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aryanchauhan31-blue?logo=linkedin)](https://linkedin.com/in/aryanchauhan31)
[![GitHub](https://img.shields.io/badge/GitHub-aryanchauhan31-black?logo=github)](https://github.com/aryanchauhan31)
[![Email](https://img.shields.io/badge/Email-ac11274@nyu.edu-red?logo=gmail)](mailto:ac11274@nyu.edu)

---

## Technical Stack

Languages: Python (vLLM, LlamaIndex, PyTorch, DeepSpeed, NumPy, Scikit-Learn, PySpark, TensorFlow), CUDA C++, SQL
Domains: LLMs, RAG, Inference Optimization (Paged Attention, Chunked Prefix), Vision-Language Models, Quantization (bfloat16, AWQ, QAT), Distributed Training (FSDP, DDP), Phishing Detection & Security AI
Tools: Docker, Kubernetes, Slurm, Hugging Face Transformers, LangChain, Ollama, GCP (Vertex AI, BigQuery), AWS, Apache (Spark, Airflow, Beam), Weights & Biases, CI/CD Pipelines

---

## Specializations

- **Inference & Model Optimization**
  - vLLM & Serving: Chunked Prefix Caching, Paged Attention, Flash Attention, Disaggregated Serving, Tensor Parallelism
  - Quantization: Activation-aware Weight Quantization, SmoothQuant, Quantization-Aware Training (QAT), Dynamic Quantization, Native bfloat16 Optimization
  - Techniques: Key-Value (KV) Caching, JIT Compilation, CUDA Kernels

- **Parameter-Efficient Fine-Tuning (PEFT) & Pruning**
  - PEFT Methods: LoRA, QLoRA, Soft Prompting (PEFT library)
  - Pruning Strategies: NetAdapt-style structured pruning, Magnitude pruning, Filter/Channel pruning
  - Optimization: Latency-aware model compression (e.g., 64% size reduction on DistilBERT) using Hugging Face Optimum

- **Distributed Training & MLOps**
  - Parallelism: Fully Sharded Data Parallel (FSDP), PyTorch DDP, Tensor Parallelism
  - Memory Efficiency: DeepSpeed (ZeRO-Offload), Gradient Checkpointing, Mixed Precision Training
  - Infrastructure: Multi-node cluster scaling (Slurm/Kubernetes), Docker containerization

- **Multimodal & RAG Architectures**
  - RAG Systems: LlamaIndex-based pipelines, Vector Search, Knowledge Retrieval, Faithfulness Evaluation
  - Vision-Language: CLIP-style ViT+BERT alignment, Cross-modal Knowledge Distillation
  - Architectures: ResNet + DistilBERT student-teacher networks, Transformer-based encoders

---

##  Current Work
AI Engineer — Charlemagne Labs (Jan 2026 – Present)
Building real-time phishing URL classification systems using small language models:

Fine-tuning a Gemma 3-based 270M-parameter SLM, improving precision/recall from 49%/90% to 96%/94% across 30+ model iterations
Architected a teacher–student evaluation framework using a 9B-parameter LLM to identify and fix SLM weaknesses
Engineered a multi-stage synthetic data pipeline producing 36K balanced training samples for robustness against unseen threats
Deployed continuous evaluation on AWS against OpenPhish and Tranco ground-truth datasets

---
## Featured Projects
### CUDA Kernel Library
A collection of GPU-accelerated kernels written in CUDA C++ for core ML and numerical operations:

Matrix Multiplication — tiled shared-memory matmul and sparse matmul (CSR format)
Parallel Reductions — 4 progressive optimization stages (naive → warp-level)
Prefix Sum — Blelloch-style inclusive/exclusive scan
Dot Product & Vector Add — fundamental CUDA parallelism patterns
CUDA Streams — overlapping computation with memory transfers
Categorical Cross-Entropy — custom loss kernel for training pipelines

### 🤗 Hugging Face Transformers

- **[#38487: Enable `device_map="auto"` support for Dinov2](https://github.com/huggingface/transformers/pull/38487)**  
  Enabled automatic device placement for Dinov2 by defining `_no_split_modules`, unlocking inference across CPU/GPU seamlessly.

- **[#38461: Add `GLPNImageProcessorFast`](https://github.com/huggingface/transformers/pull/38461)**  
  Implemented a fast image processor variant for the GLPN model using TorchVision. Achieved functional parity with the original (max abs diff < 1e-7) and added complete tests.

- **[#38509: SparseVLM – Visual Token Sparsification for Efficient VLM Inference](https://github.com/huggingface/transformers/issues/38509)**  
  Proposed support for SparseVLM: a **training-free**, plug-and-play method to prune redundant image tokens in VLMs like BLIP and Flamingo.  
  It uses **attention-guided token selection and recycling** for up to **60% FLOPs reduction** with minimal accuracy loss. Currently preparing an implementation compatible with 🤗 `transformers`.


### End-to-End RAG System for Document Q&A

LlamaIndex + Hugging Face pipeline achieving 94% faithfulness and 89% answer relevancy
vLLM integration with Chunked Prefix Caching for 3× decoding throughput improvement
Native bfloat16 on A100 GPUs to eliminate quantization precision loss

## Multimodal VQA Optimization

ViT+BERT cross-modal architecture with ResNet18+DistilBERT student model (60% compression)
QAT + DDP training across 4×L4 GPUs — 1.8× speed-up, final loss of 0.098

## DistilBERT Compression Pipeline

64% model size reduction (268MB → 96MB) preserving 98.9% GLUE accuracy
Automated fine-tuning and dynamic quantization via Hugging Face Optimum

---

## 📫 Contact
- 📧 Email: [ac11274@nyu.edu](mailto:ac11274@nyu.edu)
- 🌐 Portfolio: [github.com/aryanchauhan31](https://github.com/aryanchauhan31)

---

*Let’s build something impactful together.*
