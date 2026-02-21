> _Repository for WRV/AISysProj/ws2425/a2.5-classify-math-publications/team820_

## **Topic:** WS2425 Assignment 2.5: Classify Math Publications

## Overview

This project implements a hierarchical transformer model for arXiv document classification using mathematical formulas, featuring:

-   **Math-Aware BPE Tokenization** (7.5k vocabulary with Unicode symbols)
-   **Dual Transformer Architecture** (Formula + Document level processing)
-   **CLS Token Pooling** for hierarchical embeddings
-   **Curriculum Learning** with cosine LR scheduling

---

## Solution Summary

Solution summary is available in **[Solution_Summary.pdf](Solution_Summary.pdf)**.

---

## Key Features

### Formula Processing

-   **XML-Aware Tokenization**
    -   Sanitizes the XML tags by removing their attributes, and does not breaks a tag rather considers it as a standalone token.
-   **Number Tokenization**
    -   Tokenizes the number in consistent manner. (e.g., 10.34 tokenized into [INT][FLOAT])
-   **Hybrid Vocabulary**
    -   Base tokens: XML tags + common math symbols
    -   Learned BPE merges: 7.5k total vocabulary

### Model Architecture

-   **Hierarchical Encoding**
    ```math
    h_{\text{formula}} = \text{Transformer}_{\text{formula}}([\text{CLS}] \oplus \{h_{\text{token}}^i\}_{i=1}^{350}) \\
    h_{\text{doc}} = \text{Transformer}_{\text{doc}}([\text{CLS}] \oplus \{h_{\text{formula}}^i\}_{i=1}^{10})
    ```
-   **Dual CLS Tokens**
    -   Formula-level CLS: 256-dim embedding
    -   Document-level CLS: 256-dim embedding

### Training Optimization

-   **Dynamic Padding**
    -   Formula level: 350 tokens max
    -   Document level: 10 formulas max
-   **Gradient Clipping** (‖g‖ ≤ 1.0)

---

## Dependencies

### Core Requirements

-   **PyTorch 2.6+** (CUDA 12.4 with cudnn9 recommended)
-   **Tokenizers** (Hugging Face implementation)
-   **MLflow** (Experiment tracking)

---

### **Prerequisites**

Ensure that you have the following installed on your system:

-   [Docker](https://docs.docker.com/get-docker/)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

To verify that Docker is installed, run:

```bash
docker --version
```

If you want to use GPU support, ensure the NVIDIA runtime is set up by running:

```bash
docker run --rm --gpus all nvidia/cuda:11.7.1-base nvidia-smi
```

This should display information about your GPU.

---

### **Building the Docker Image**

Clone your repository and navigate to the project directory:

```bash
git clone https://gitlab.rrze.fau.de/wrv/AISysProj/ws2425/a2.5-classify-math-publications/team820.git
cd team820
```

Then, build the Docker image using:

```bash
docker build -f Dockerfile -t ai-sys-img .
```

This will create an image named `ai-sys-img`.

---

### **Running the MLflow Server Inside a Docker Container**

To start the MLflow server inside a Docker container and expose the MLflow UI on port 5000, run:

```bash
docker run -d --gpus all -p 5000:5000 -v $(pwd)/mlruns:/workspace/mlruns -w /workspace --name mlflow_server ai-sys-img mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /workspace/mlruns
```

Once the server is running, you can access the MLflow UI at:

```
http://localhost:5000
```

---

### **Dataset Preparation**

Run the dataset preparation script:

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace --network="host" ai-sys-img python data/data_processing.py
```

---

### Training

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace --network="host" ai-sys-img python train.py
```

---

### Evaluation

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace --network="host" ai-sys-img python evaluate.py Data/example-test-results.json Data/my_test_result.json

```

### Run the Agent in Server

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace --network="host" ai-sys-img python server_interaction.py
```

---

## Repository Structure

```bash
project-root
├── assignment-2.5-guide.pdf
├── assignment-2.5.pdf
├── category-discripency.txt
├── ckpts
│   ├── best_model.pth
│   ├── curr_best_model.pth
│   └── last_trained_best_model.pth
├── data
│   ├── data_processing.py
│   └── math_symbol_extractor.py
├── Dockerfile
├── evaluate.py
├── generate_result.py
├── mlruns
├── model.py
├── README.md
├── server-config.json
├── server_interaction.py
├── Solution_Summary.pdf
└── train.py
```

---

## Important Notes

-   **arXiv Categories**  
    18 primary classes including math, physics, astro-ph, hep-ph, cs etc.
-   **Formula Limits**  
    Documents with > 10 formulas are truncated to 10
-   **Tokenization**  
    Special handling for 750+ Unicode math symbols via explicit symbol list

---

### Contact

Maintainer: Redwanul Karim  
Email: [redwanul.karim@fau.de]
