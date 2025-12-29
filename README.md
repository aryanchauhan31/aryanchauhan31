# 👋 Hi, I'm Aryan 

I'm a graduate student at **NYU** pursuing a Master's in Computer Engineering, passionate about building efficient and scalable AI systems. I focus on **LLM optimization**, **multimodal models**, and **open-source contributions**—most recently to the 🤗 `transformers` library.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aryanchauhan31-blue?logo=linkedin)](https://linkedin.com/in/aryanchauhan31)
[![GitHub](https://img.shields.io/badge/GitHub-aryanchauhan31-black?logo=github)](https://github.com/aryanchauhan31)
[![Email](https://img.shields.io/badge/Email-ac11274@nyu.edu-red?logo=gmail)](mailto:ac11274@nyu.edu)

---

## 🛠️ Technical Stack

- **Languages**: Python (vLLM, LlamaIndex, PyTorch, DeepSpeed, NumPy, Scikit-Learn, PySpark, TensorFlow), CUDA C++, C/C++, SQL
- **Domains**: LLMs, RAG, Inference Optimization (Paged Attention, Chunked Prefix), Vision-Language Models, Quantization (bfloat16, QAT), Distributed Training (FSDP, DDP), Recommender Systems
- **Tools**: Docker, Kubernetes, Slurm, Hugging Face Transformers, LangChain, Ollama, GCP (Vertex AI), AWS, Apache Spark, Airflow, Weights & Biases

---

## 🧠 Specializations

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

## 🚀 Recent Work



### 🤗 Hugging Face Transformers

- **[#38487: Enable `device_map="auto"` support for Dinov2](https://github.com/huggingface/transformers/pull/38487)**  
  Enabled automatic device placement for Dinov2 by defining `_no_split_modules`, unlocking inference across CPU/GPU seamlessly.

- **[#38461: Add `GLPNImageProcessorFast`](https://github.com/huggingface/transformers/pull/38461)**  
  Implemented a fast image processor variant for the GLPN model using TorchVision. Achieved functional parity with the original (max abs diff < 1e-7) and added complete tests.

- **[#38509: SparseVLM – Visual Token Sparsification for Efficient VLM Inference](https://github.com/huggingface/transformers/issues/38509)**  
  Proposed support for SparseVLM: a **training-free**, plug-and-play method to prune redundant image tokens in VLMs like BLIP and Flamingo.  
  It uses **attention-guided token selection and recycling** for up to **60% FLOPs reduction** with minimal accuracy loss. Currently preparing an implementation compatible with 🤗 `transformers`.

---

### 📸 Multimodal VQA Optimization
- Developed ViT+BERT architecture for Visual Question Answering.
- Trained with QAT + DDP over 4×L4 GPUs for 1.8× speed-up and 60% model compression.

### 🧠 DistilBERT Compression Pipeline
- Reduced size by 64% with dynamic quantization.
- Automated finetuning and benchmarking via Hugging Face tools.

---

## 📈 What I'm Looking For
I'm currently open to:
- Full-time roles in Machine Learning Engineering or AI Infrastructure, specifically focused on LLM Efficiency, Inference Optimization, and Distributed Training.
- Opportunities to engineer scalable RAG pipelines and deploy vLLM-based serving architectures in production environments.

---

## 📫 Contact
- 📧 Email: [ac11274@nyu.edu](mailto:ac11274@nyu.edu)
- 🌐 Portfolio: [github.com/aryanchauhan31](https://github.com/aryanchauhan31)

---

*Let’s build something impactful together.*
