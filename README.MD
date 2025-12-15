# 🚀 Gemini Study Jams 25

Welcome to **Gemini Study Jams 25** - A comprehensive 3-week program designed to master advanced AI model fine-tuning and deployment techniques using Google's cutting-edge **Gemini** and **Gemma** models.

[![Google AI](https://img.shields.io/badge/Google-AI-4285F4?logo=google)](https://ai.google.dev/)
[![Gemini](https://img.shields.io/badge/Gemini-Models-00897B)](https://deepmind.google/technologies/gemini/)

---

## 📚 Table of Contents

- [About](#-about)
- [Program Overview](#-program-overview)
- [Repository Structure](#-repository-structure)
- [Session Details](#-session-details)
  - [Week 1: Supervised Fine-Tuning](#week-1-supervised-fine-tuning-in-practice)
  - [Week 2: Parameter Efficient Fine-Tuning (PEFT)](#week-2-fine-tuning-open-models)
  - [Week 3: Building & Deploying AI Apps](#week-3-building--deploying-ai-apps)
- [Prerequisites](#-prerequisites)
- [Getting Started](#-getting-started)
- [Learning Objectives](#-learning-objectives)
- [Resources](#-resources)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About

This repository contains all the materials, notebooks, presentations, and resources for the **Gemini Study Jams 25** program. The study jam focuses on practical implementations of:

- **Supervised Fine-Tuning (SFT)** with Gemini models
- **Parameter Efficient Fine-Tuning (PEFT)** using LoRA
- **Model Deployment** and MLOps practices
- **Building production-ready AI applications**

Whether you're new to fine-tuning or looking to level up your AI engineering skills, this program provides hands-on experience with state-of-the-art techniques.

---

## 📅 Program Overview

The Gemini Study Jams is structured as a **3-week intensive learning experience**, with each week focusing on specific aspects of AI model fine-tuning and deployment:

| Week | Focus Area | Key Topics |
|------|-----------|------------|
| **Week 1** | Supervised Fine-Tuning | Gemini SFT basics, Image captioning |
| **Week 2** | PEFT with LoRA | Parameter-efficient methods, Gemma fine-tuning |
| **Week 3** | Deployment & MLOps | Building AI apps, Production deployment |

---

## 📂 Repository Structure

```
Gemini-Study-Jams-25/
├── README.MD                                          # This file
├── syllabus.MD                                        # Detailed syllabus
├── gemini-study-jams-introductory-session.pdf        # Introductory materials
│
├── Session1 - Fine Tuning Basics & SFT/
│   ├── RESOURCES.MD                                   # Session 1 resources
│   ├── Session-1 SFT.ipynb                           # Hands-on notebook
│   └── Session 1 - Fine Tuning Basics & Supervised Fine Tuning.pdf
│
└── Session 2 - Supervised PEFT Using LoRA/
    ├── RESOURCES.MD                                   # Session 2 resources
    ├── Session_2_PEFT_with_LoRA.ipynb                # LoRA implementation
    └── Session 2 - Supervised PEFT Using LoRA.pdf
```

---

## 🗓️ Session Details

### Week 1: Supervised Fine-Tuning in Practice

**Focus**: Learn the fundamentals of supervised fine-tuning with Gemini models

#### 📖 What You'll Learn
- Core concepts of supervised fine-tuning
- Hands-on practice with Gemini models
- Specialized applications for computer vision tasks
- Real-world image captioning scenarios

#### 📁 Session Materials
- **[Session-1 SFT.ipynb](Session1%20-%20Fine%20Tuning%20Basics%20%26%20SFT/Session-1%20SFT.ipynb)** - Interactive fine-tuning notebook
- **[RESOURCES.MD](Session1%20-%20Fine%20Tuning%20Basics%20%26%20SFT/RESOURCES.MD)** - Comprehensive learning resources
- **Session PDF** - Theoretical foundations and slides

#### 🔗 Learning Resources
- [Supervised Fine-tuning for Gemini](https://www.cloudskillsboost.google/course_templates/1368)
- [Supervised Fine-tuning with Gemini for Image Captioning](https://www.skills.google/focuses/124952)

---

### Week 2: Fine-tuning Open Models

**Focus**: Explore advanced fine-tuning techniques with Gemma models using LoRA

#### 📖 What You'll Learn
- Parameter Efficient Fine-Tuning (PEFT) concepts
- LoRA (Low-Rank Adaptation) architecture and implementation
- Reducing trainable parameters from billions to millions
- Practical implementation with Qwen 2.5-3B model
- Memory optimization techniques with QLoRA

#### 💻 Hands-on Project
- **Case Study**: Modern-to-Shakespearean translation
- **Dataset**: 50 instruction-response pairs
- **Results**: 99.88% parameter reduction with maintained performance

#### 📁 Session Materials
- **[Session_2_PEFT_with_LoRA.ipynb](Session%202%20-%20Supervised%20PEFT%20Using%20LoRA/Session_2_PEFT_with_LoRA.ipynb)** - LoRA implementation notebook
- **[RESOURCES.MD](Session%202%20-%20Supervised%20PEFT%20Using%20LoRA/RESOURCES.MD)** - Advanced PEFT resources
- **Session PDF** - LoRA theory and best practices

#### 🔗 Learning Resources
- [Gemma Fine-tuning Documentation](https://ai.google.dev/gemma/docs/tune)
- [Fine-tune Gemma models in Keras using LoRA](https://ai.google.dev/gemma/docs/lora_tuning)
- [Full Model Fine-Tune using Hugging Face Transformers](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune)

#### 🔑 Key Research Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

### Week 3: Building & Deploying AI Apps

**Focus**: Transform your fine-tuned models into production-ready applications

#### 📖 What You'll Learn
- AI-powered code generation with Gemini Code Assist
- Building intelligent agents and chatbots
- Production deployment strategies
- Model lifecycle management
- Model performance monitoring and evaluation

#### 🔗 Learning Resources

**Application Development**:
- [Build Apps with Gemini Code Assist](https://www.cloudskillsboost.google/focuses/104240)
- [Codelab: Agent Development Kit](https://google.github.io/adk-docs/get-started/quickstart/)

**MLOps & Production**:
- [Machine Learning Operations (MLOps) for Generative AI](https://www.cloudskillsboost.google/paths/1283/course_templates/927)
- [MLOps with Vertex AI: Model Evaluation](https://www.cloudskillsboost.google/paths/1283/course_templates/1080)

---

## 📋 Prerequisites

Before starting the Gemini Study Jams, you should have:

- ✅ **Basic understanding of machine learning concepts**
  - Neural networks fundamentals
  - Training and evaluation processes
  
- ✅ **Familiarity with Python programming**
  - Python 3.7+ experience
  - Comfortable with NumPy, pandas
  
- ✅ **Development environment setup**
  - Google Colab account (recommended)
  - OR local Python environment with GPU access
  
- ✅ **Enthusiasm for cutting-edge AI technology!** 🚀

### Recommended Background
- Experience with deep learning frameworks (PyTorch or TensorFlow)
- Understanding of transformer architectures (helpful but not required)
- Basic knowledge of NLP or computer vision concepts

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AnshuSharma111/Gemini-Study-Jams-25.git
cd Gemini-Study-Jams-25
```

### 2. Set Up Your Environment

#### Option A: Google Colab (Recommended)
- Open the notebooks directly in Google Colab
- No local setup required!
- Free GPU access for training

#### Option B: Local Setup
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install transformers peft accelerate bitsandbytes
pip install datasets evaluate
pip install torch torchvision  # Install PyTorch for your system
```

### 3. Start Learning!

1. **Begin with Week 1**: Review the [syllabus.MD](syllabus.MD)
2. **Read the resources**: Check each session's RESOURCES.MD file
3. **Run the notebooks**: Follow along with the interactive notebooks
4. **Practice**: Complete the exercises and challenges
5. **Experiment**: Modify parameters and try your own datasets

---

## 🎯 Learning Objectives

By the end of this program, you will be able to:

- ✅ **Master supervised fine-tuning techniques** for Gemini models
- ✅ **Understand and implement** advanced optimization methods like LoRA and QLoRA
- ✅ **Build and deploy** production-ready AI applications
- ✅ **Implement MLOps best practices** for generative AI
- ✅ **Optimize model performance** while managing computational resources
- ✅ **Evaluate and monitor** model quality in production environments

---

## 📚 Resources

### Official Documentation
- [Google AI Gemini Documentation](https://ai.google.dev/gemini-api/docs)
- [Gemma Models Documentation](https://ai.google.dev/gemma/docs)
- [Hugging Face PEFT Library](https://huggingface.co/docs/peft/)

### Key Libraries Used
- **[Transformers](https://github.com/huggingface/transformers)** - State-of-the-art NLP models
- **[PEFT](https://github.com/huggingface/peft)** - Parameter Efficient Fine-Tuning
- **[Accelerate](https://github.com/huggingface/accelerate)** - Distributed training made easy
- **[BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)** - Quantization for efficiency

### Research Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

### Community
- [Hugging Face Community Forum](https://discuss.huggingface.co/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Google Cloud Skills Boost](https://www.cloudskillsboost.google/)

---

## 🤝 Contributing

We welcome contributions to improve the learning materials! Here's how you can help:

1. **Fork** the repository
2. **Create** a new branch (`git checkout -b feature/improvement`)
3. **Make** your changes
4. **Commit** your changes (`git commit -am 'Add new resource'`)
5. **Push** to the branch (`git push origin feature/improvement`)
6. **Create** a Pull Request

### Contribution Ideas
- Add new practice exercises
- Improve documentation
- Share your fine-tuned models
- Report issues or bugs
- Suggest new resources

---

## 📄 License

This project is open for educational purposes. Please check with the repository maintainer for specific licensing terms.

---

## 🙏 Acknowledgments

- **Google AI** for the Gemini and Gemma models
- **Hugging Face** for the transformers and PEFT libraries
- **All contributors** and participants of Gemini Study Jams 25

---

## 📞 Contact & Support

- **Issues**: Please use [GitHub Issues](https://github.com/AnshuSharma111/Gemini-Study-Jams-25/issues) for bug reports or feature requests
- **Discussions**: Join the community discussions in the repository

---

<div align="center">

**Happy Learning! 🧠✨**

*Master the art of fine-tuning and build the future of AI applications*

[⬆ Back to Top](#-gemini-study-jams-25)

</div>
