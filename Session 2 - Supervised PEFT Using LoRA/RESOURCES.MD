# 🎯 Session 2: Parameter Efficient Fine-Tuning (PEFT) with LoRA

Welcome to Session 2! This session focuses on **Parameter Efficient Fine-Tuning (PEFT)** techniques, specifically **LoRA (Low-Rank Adaptation)**, which dramatically reduces the computational requirements for fine-tuning large language models.

---

## 🔍 What You'll Learn

- **Parameter Efficiency**: Reduce trainable parameters from billions to millions
- **LoRA Architecture**: Understanding low-rank matrix decomposition
- **Practical Implementation**: Hands-on LoRA fine-tuning with Qwen 2.5-3B
- **Evaluation Metrics**: Measuring model performance and efficiency gains

---

## 📚 Core Research Papers

### 🏆 Foundational Papers
- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**
  - **Authors**: Edward Hu, Yelong Shen, et al. (Microsoft Research)
  - **Key Innovation**: Reduces trainable parameters by 10,000x while maintaining performance
  - **Must-read**: Understanding the mathematical foundation of LoRA

- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)**
  - **Authors**: Tim Dettmers, et al. (University of Washington)
  - **Innovation**: Combines 4-bit quantization with LoRA for extreme efficiency
  - **Impact**: Fine-tune 65B models on a single GPU

- **[AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)**
  - **Key Concept**: Dynamic rank allocation across different layers
  - **Advantage**: Better parameter utilization than fixed-rank LoRA

### 🔬 Advanced PEFT Methods
- **[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)**
  - Alternative PEFT approach using learnable prefix tokens
  - Comparison baseline for LoRA effectiveness

- **[P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning](https://arxiv.org/abs/2110.07602)**
  - Deep prompt tuning across all layers
  - Comprehensive PEFT method comparison

---

## 🎥 Essential Video Resources

### 🎓 Technical Deep Dives
- **[Yannic Kilcher: LoRA Paper Explained](https://www.youtube.com/watch?v=DhRoTONcyZE)**
  - In-depth paper walkthrough with mathematical intuition
  - Visual explanations of low-rank decomposition

- **[Andrej Karpathy: The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)**
  - Essential background for understanding parameter updates
  - Foundation knowledge for PEFT techniques

- **[Hugging Face: PEFT Library Tutorial](https://www.youtube.com/watch?v=Us5ZFp16PaU)**
  - Practical implementation guide
  - Multiple PEFT methods comparison

### 🔍 Mathematical Foundations
- **[3Blue1Brown: Linear Algebra - Matrix Decomposition](https://www.youtube.com/watch?v=P2LTAUO1TdA)**
  - SVD and low-rank approximations
  - Mathematical intuition behind LoRA

- **[StatQuest: Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=FgakZw6K1QQ)**
  - Dimensionality reduction concepts
  - Relates to LoRA's rank reduction approach

---

## 💻 Hands-On Tutorials & Code

### 🛠️ Official Implementations
- **[Hugging Face PEFT Library](https://github.com/huggingface/peft)**
  - Official PEFT implementation with LoRA, AdaLoRA, QLoRA
  - Production-ready code with extensive examples

- **[Microsoft LoRA Implementation](https://github.com/microsoft/LoRA)**
  - Original LoRA implementation from the paper authors
  - Research-grade code with detailed documentation

### 📓 Interactive Notebooks
- **[Google Colab: LoRA Fine-tuning Tutorial](https://colab.research.google.com/github/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq.ipynb)**
  - Step-by-step LoRA implementation
  - Sequence-to-sequence fine-tuning example

- **[Kaggle: QLoRA Implementation Guide](https://www.kaggle.com/code/jhoward/getting-started-with-qlora)**
  - Efficient 4-bit quantization + LoRA
  - Memory-optimized fine-tuning

### 🎯 Practical Projects
- **[Fine-tune Llama 2 with LoRA](https://github.com/facebookresearch/llama-recipes/tree/main/recipes/finetuning)**
  - Production-scale LoRA fine-tuning
  - Multi-GPU training strategies

---

## 🔧 Tools & Libraries

### 📦 Essential Libraries
```bash
# Core PEFT Libraries
pip install peft transformers accelerate
pip install bitsandbytes  # For QLoRA quantization
pip install datasets      # Dataset handling
pip install evaluate      # Model evaluation metrics
```

### 🏗️ Development Stack
- **[PEFT](https://github.com/huggingface/peft)** - Parameter Efficient Fine-Tuning methods
- **[Transformers](https://github.com/huggingface/transformers)** - Model architectures and utilities
- **[Accelerate](https://github.com/huggingface/accelerate)** - Distributed training made simple
- **[BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)** - 8-bit and 4-bit model quantization

### 🖥️ Hardware Requirements
- **Minimum**: 8GB GPU VRAM for small models (3B parameters)
- **Recommended**: 16GB+ GPU VRAM for medium models (7B parameters)
- **QLoRA**: Enables fine-tuning 13B+ models on consumer hardware

---

## 📊 Understanding LoRA Hyperparameters

### 🎚️ Key Parameters
- **`r` (Rank)**: Controls the bottleneck dimension
  - Typical values: 4, 8, 16, 32, 64
  - Higher rank = more parameters but better capacity
  
- **`lora_alpha`**: Scaling factor for LoRA weights
  - Controls the magnitude of LoRA adaptations
  - Common practice: `lora_alpha = 2 * r`

- **`target_modules`**: Which layers to apply LoRA
  - Common choices: `["q_proj", "v_proj", "o_proj"]`
  - More modules = more parameters but better adaptation

### 📈 Parameter Efficiency Examples
```python
# Standard Fine-tuning: 3.089B parameters (100%)
# LoRA (r=16): 3.69M parameters (0.12%)
# QLoRA (r=16, 4-bit): 0.92M parameters (0.03%)
```

---

## 🔍 Evaluation & Benchmarks

### 📏 Key Metrics
- **Perplexity**: Language modeling performance measure
- **BLEU Score**: Translation and generation quality
- **Parameter Efficiency**: Trainable params / Total params ratio
- **Memory Usage**: Peak GPU memory during training

### 🏆 Benchmark Datasets
- **GLUE**: General Language Understanding tasks
- **SuperGLUE**: Advanced language understanding benchmarks  
- **HellaSwag**: Commonsense reasoning evaluation
- **HumanEval**: Code generation assessment

### 📊 Evaluation Tools
```python
# Perplexity calculation
import math
train_loss = 2.245  # From our Shakespearean example
perplexity = math.exp(train_loss)  # ≈ 9.43
```

---

## 🎭 Case Study: Shakespearean Translation

Our notebook demonstrates LoRA fine-tuning for modern-to-Shakespearean translation:

### 📋 Dataset Characteristics
- **Size**: 50 instruction-response pairs
- **Task**: Style transfer (modern → Shakespearean English)
- **Format**: Instruction-following with clear input/output structure

### 🔧 Training Configuration
```python
# LoRA Configuration
r=4                           # Low rank for efficiency
lora_alpha=16                # 4x scaling factor
target_modules=["q_proj", "v_proj", "o_proj"]  # Attention layers only

# Training Settings  
batch_size=4                 # Per device
gradient_accumulation=2      # Effective batch size: 8
learning_rate=2e-4          # Higher LR for PEFT
epochs=5                    # Quick convergence
```

### 📈 Results Analysis
- **Training Loss**: 2.245 → **Perplexity**: 9.43
- **Parameter Reduction**: 99.88% fewer trainable parameters
- **Style Consistency**: Successfully learns Shakespearean patterns

---

## 🚨 Common Challenges & Solutions

### ⚡ Memory Issues
- **Problem**: GPU OOM errors during training
- **Solution**: Use gradient checkpointing, smaller batch sizes, QLoRA quantization

### 🎯 Poor Adaptation
- **Problem**: Model doesn't learn task-specific behaviors
- **Solution**: Increase rank `r`, target more modules, check learning rate

### 📉 Catastrophic Forgetting  
- **Problem**: Model loses general capabilities
- **Solution**: Lower learning rates, fewer training epochs, regularization

### 🔄 Hyperparameter Tuning
```python
# Systematic hyperparameter search
ranks = [4, 8, 16, 32]
alphas = [8, 16, 32, 64] 
learning_rates = [1e-4, 2e-4, 5e-4]
```

---

## 🔬 Advanced Topics

### 🎭 Multi-Task LoRA
- Train separate LoRA adapters for different tasks
- Switch adapters at inference time
- Modular fine-tuning approach

### 🔄 LoRA Composition
- Combine multiple LoRA adapters
- Arithmetic operations on adapter weights
- Task interpolation and blending

### 🎯 Targeted Fine-Tuning
```python
# Different ranks for different layer types
peft_config = LoraConfig(
    r=16,
    target_modules={
        "attention": ["q_proj", "v_proj"],  # Higher capacity
        "mlp": ["gate_proj"]                # Lower capacity
    }
)
```

---

## 📖 Supplementary Reading

### 📰 Blog Posts & Tutorials
- **[Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/)**
  - Comprehensive guide to all PEFT methods
  - Best practices and optimization tips

- **[Sebastian Ruder: Parameter-Efficient Fine-Tuning](https://ruder.io/parameter-efficient-fine-tuning/)**
  - Survey of PEFT methods beyond LoRA
  - Comparative analysis and recommendations

- **[Towards Data Science: LoRA Explained](https://towardsdatascience.com/lora-low-rank-adaptation-of-large-language-models-7f981b26d8b4)**
  - Mathematical intuition and practical examples
  - Implementation details and optimization tips

### 🎪 Community Resources
- **[r/MachineLearning LoRA Discussions](https://www.reddit.com/r/MachineLearning/search/?q=LoRA)**
- **[Hugging Face Community Forum - PEFT](https://discuss.huggingface.co/c/peft/)**
- **[Papers With Code - Parameter Efficient Fine-Tuning](https://paperswithcode.com/task/parameter-efficient-fine-tuning)**

---

## 🎯 Session Objectives

### ✅ Learning Checklist
- [ ] Understand LoRA mathematical foundations
- [ ] Implement LoRA fine-tuning with PEFT library
- [ ] Compare parameter efficiency vs. full fine-tuning
- [ ] Evaluate model performance with appropriate metrics
- [ ] Experiment with hyperparameter optimization
- [ ] Handle memory constraints with QLoRA

### 🏆 Practical Exercises
1. **Reproduce Notebook Results**: Run the Shakespearean translation example
2. **Hyperparameter Tuning**: Experiment with different `r` and `lora_alpha` values
3. **Target Module Analysis**: Compare different `target_modules` configurations
4. **Custom Dataset**: Apply LoRA to your own fine-tuning task
5. **Memory Optimization**: Implement QLoRA for larger models

### 🔍 Advanced Challenges
- Implement multi-task LoRA adapters
- Compare LoRA vs. other PEFT methods (Prefix-tuning, P-tuning v2)
- Optimize for inference speed and memory usage
- Develop custom evaluation metrics for your specific task

---

## 📁 Session Materials

### 📋 Files in This Session
- **[Session_2_PEFT_with_LoRA.ipynb](Session_2_PEFT_with_LoRA.ipynb)** - Interactive LoRA implementation
- **[Session 2 - Supervised PEFT Using LoRA.pdf](Session%202%20-%20Supervised%20PEFT%20Using%20LoRA.pdf)** - Theoretical foundations and slides

### 💾 Code Snippets Repository
```bash
# Quick start commands
git clone https://github.com/huggingface/peft
cd peft/examples
python run_lora_finetuning.py --config examples/configs/lora_config.json
```

---

*Ready to master parameter-efficient fine-tuning? Start with the foundational papers, then dive into the hands-on notebook! 🚀*

## 🔗 Quick Navigation
- [Core Papers](#-core-research-papers) | [Videos](#-essential-video-resources) | [Code](#-hands-on-tutorials--code) | [Tools](#-tools--libraries) | [Evaluation](#-evaluation--benchmarks) | [Case Study](#-case-study-shakespearean-translation)