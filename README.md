# 👋 Hi, I'm Aryan 

I'm a graduate student at **NYU** pursuing a Master's in Computer Engineering, passionate about building efficient and scalable AI systems. I focus on **LLM optimization**, **multimodal models**, and **open-source contributions**—most recently to the 🤗 `transformers` library.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aryanchauhan31-blue?logo=linkedin)](https://linkedin.com/in/aryanchauhan31)
[![GitHub](https://img.shields.io/badge/GitHub-aryanchauhan31-black?logo=github)](https://github.com/aryanchauhan31)
[![Email](https://img.shields.io/badge/Email-ac11274@nyu.edu-red?logo=gmail)](mailto:ac11274@nyu.edu)

---

## 🛠️ Technical Stack

- **Languages**: Python (PyTorch, DeepSpeed, NumPy, Scikit-Learn, PySpark, TensorFlow), CUDA C++,C/C++, SQL 
- **Domains**: LLMs, Vision-Language Models, Quantization, Distributed Training (DDP), Recommender Systems
- **Tools**: Docker, Slurm, Hugging Face Transformers, LangChain, Ollama, GCP, AWS, Spark, Airflow

---

## 🧠 Specializations

- **Quantization Techniques**
  - SmoothQuant, Dynamic Quantization, Quantization-Aware Training (QAT)
  - Frameworks: PyTorch FX, ONNX Runtime, Hugging Face Optimum

- **Pruning Strategies**
  - Filter/channel pruning, magnitude pruning, **NetAdapt-style structured pruning**
  - Latency-aware model slimming via FLOPs/accuracy trade-offs

- **Distributed Training**
  - PyTorch Distributed Data Parallel (DDP), Deepspeed
  - Mixed precision (FP16), gradient accumulation, multi-node cluster scaling

- **Multimodal Systems**
  - CLIP-like ViT-BERT architectures
  - Vision-Language alignment, CLIPScore evaluation, knowledge distillation

> 🛠️ Comfortable implementing research papers from scratch and profiling performance with tools like Weights & Biases and PyTorch Profiler.

---

## 🚀 Recent Work



### 🤗 Hugging Face Transformers

- **[#38461: Enable `device_map="auto"` support for Dinov2](https://github.com/huggingface/transformers/pull/38461)**  
  Enabled automatic device placement for Dinov2 by defining `_no_split_modules`, unlocking inference across CPU/GPU seamlessly.

- **[#38285: Add `GLPNImageProcessorFast`](https://github.com/huggingface/transformers/pull/38285)**  
  Implemented a fast image processor variant for the GLPN model using TorchVision. Achieved functional parity with the original (max abs diff < 1e-7) and added complete tests.

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
- Remote internships or research collaborations in **LLM efficiency**, **model compression**, or **AI infrastructure**
- Open-source projects focused on cutting-edge ML research

---

## 📫 Contact
- 📧 Email: [ac11274@nyu.edu](mailto:ac11274@nyu.edu)
- 🌐 Portfolio: [github.com/aryanchauhan31](https://github.com/aryanchauhan31)

---

*Let’s build something impactful together.*
