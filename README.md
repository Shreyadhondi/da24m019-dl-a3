# DA6401 Assignment 3

# Sequence-to-Sequence Transliteration using RNNs and Attention

## Student Information

* **Name:** Shreya Dhondi
* **Roll Number:** DA24M019
* **Course:** DA6401 – Deep Learning
* **Institute:** IIT Madras

---

# Assignment Overview

This assignment focuses on building and analyzing character-level Sequence-to-Sequence (Seq2Seq) models for transliteration using recurrent neural networks.

The task involves converting Romanized words into native script using encoder-decoder architectures. I have choosen Telugu Language for this task.

The assignment includes:

* Vanilla Seq2Seq model
* Attention-based Seq2Seq model
* Hyperparameter tuning using Weights & Biases sweeps
* Evaluation on validation and test sets
* Attention heatmap visualization
* Connectivity visualization for decoder attention
* Error analysis and comparison between vanilla and attention models

---

# Dataset

The experiments were performed using the **Dakshina Dataset**.

Language used:

* **Telugu (****`te`****)**

Dataset structure used:

```text
data/te/lexicons/
```

Files used:

* `te.translit.sampled.train.tsv`
* `te.translit.sampled.dev.tsv`
* `te.translit.sampled.test.tsv`

---

# Project Structure

```text
da24m019-dl-a3/
│
├── attention_heatmaps/              # Attention heatmaps generated from test samples
│
├── checkpoints/                    # Saved model checkpoints
│   ├── best_attention.pt
│   ├── best_vanilla.pt
│   ├── global_best_attention.pt
│   ├── global_best_vanilla.pt
│
├── configs/                        # W&B sweep configurations
│   ├── sweep_attention.yaml
│   └── sweep_vanilla.yaml
│
├── predictions_attention/          # Predictions from attention model
│   └── preds.tsv
│
├── predictions_vanilla/            # Predictions from vanilla model
│   └── preds.tsv
│
├── src/
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── load_dakshina.py
│   │   └── vocab.py
│   │
│   ├── models/
│   │   ├── vanilla_seq2seq.py
│   │   ├── attention.py
│   │   └── decode.py
│   │
│   └── training/
│       ├── train_vanilla.py
│       ├── eval_vanilla.py
│       ├── train_attention.py
│       ├── eval_attention.py
│       └── metrics.py
│
├── connectivity.html               # Connectivity visualization
├── visualize_connectivity.py       # Connectivity visualization generator
│
└── README.md
```

---

# Implemented Models

## 1. Vanilla Seq2Seq Model

The baseline model consists of:

* Character Embedding Layer
* Encoder RNN / GRU / LSTM
* Decoder RNN / GRU / LSTM
* Teacher forcing during training

Supported flexibility:

* Variable embedding size
* Variable hidden size
* Multiple encoder/decoder layers
* RNN / GRU / LSTM cells
* Dropout support

---

## 2. Attention-Based Seq2Seq Model

The attention model extends the vanilla architecture by adding:

* Attention mechanism over encoder hidden states
* Dynamic alignment between source and target characters

This improves:

* Long-sequence handling
* Character alignment
* Transliteration quality

---

# Hyperparameter Sweeps

Hyperparameter optimization was performed using **Weights & Biases Sweeps**.

## Vanilla Model Sweep

Search space included:

* Embedding sizes: 16, 32, 64, 128
* Hidden sizes: 64, 128, 256
* Encoder layers: 1, 2
* Decoder layers: 1, 2
* Cell types: RNN, GRU, LSTM
* Dropout: 0.0, 0.2, 0.3
* Learning rates: 0.001, 0.0005

Approximately:

* **86+ runs** were executed for the vanilla model.

---

## Attention Model Sweep

Search space included:

* Cell types: GRU, LSTM
* Embedding sizes: 32, 64
* Hidden sizes: 128, 256
* Dropout: 0.1, 0.2, 0.3
* Learning rates: 0.0005, 0.001

Bayesian sweep strategy was used.

---

# Best Vanilla Model

## Configuration

| Hyperparameter | Value  |
| -------------- | ------ |
| Cell Type      | LSTM   |
| Embedding Size | 32     |
| Hidden Size    | 256    |
| Encoder Layers | 2      |
| Decoder Layers | 2      |
| Dropout        | 0.2    |
| Learning Rate  | 0.0005 |

## Test Performance

| Metric               | Value  |
| -------------------- | ------ |
| Test Loss            | 0.2941 |
| Token Accuracy       | 0.9162 |
| Exact Match Accuracy | 0.5432 |

Predictions stored in:

```text
predictions_vanilla/preds.tsv
```

---

# Best Attention Model

## Test Performance

| Metric               | Value  |
| -------------------- | ------ |
| Test Loss            | 0.2882 |
| Token Accuracy       | 0.8529 |
| Exact Match Accuracy | 0.5580 |

Predictions stored in:

```text
predictions_attention/preds.tsv
```

---

# Attention Heatmaps

Attention heatmaps were generated for multiple test samples.

These heatmaps visualize:

* Input character attention
* Decoder focus during output generation
* Alignment between Romanized Telugu and native Telugu script

Heatmaps are available in:

```text
attention_heatmaps/
```

---

# Connectivity Visualization

A connectivity visualization inspired by the assignment reference article was implemented.

Files related to the visualization:

```text
visualize_connectivity.py
connectivity.html
```

The visualization demonstrates:

* Which input characters the decoder attends to
* Dynamic alignment during decoding
* Character-level transliteration behavior

## How to View the Connectivity Visualization

### Method 1 (Recommended)

Open the following file directly in a browser:

```text
connectivity.html
```

### Method 2 (Using VS Code Live Server)

1. Open the repository in VS Code
2. Install the **Live Server** extension
3. Right click on `connectivity.html`
4. Click:

```text
Open with Live Server
```

This opens the animated connectivity visualization in the browser.

The visualization contains:

* Romanized input characters
* Telugu output characters
* Animated attention connections between them

The connectivity strengths represent how strongly the decoder focuses on particular input characters while generating each output character.

---

# How to Run

## 1. Create Environment

```bash
conda create -n da6401-a3 python=3.10
conda activate da6401-a3
```

---

## 2. Install Dependencies

Install all required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 3. Train Vanilla Model

```bash
python -m src.training.train_vanilla --use_wandb
```

---

## 4. Evaluate Vanilla Model

```bash
python -m src.training.eval_vanilla --ckpt_path checkpoints/global_best_vanilla.pt
```

---

## 5. Train Attention Model

```bash
python -m src.training.train_attention --use_wandb
```

---

## 6. Evaluate Attention Model

```bash
python -m src.training.eval_attention --ckpt_path checkpoints/global_best_attention.pt
```

---

# Key Observations

* LSTM models outperformed GRU and vanilla RNN models.
* Attention improved exact-match accuracy.
* Larger hidden sizes improved transliteration quality.
* Deeper encoder-decoder models performed better.
* Vanilla RNNs struggled with long-term dependencies.
* Attention helped improve alignment between source and target sequences.

---

# Files Important for Evaluation

| File / Folder            | Purpose                    |
| ------------------------ | -------------------------- |
| `train_vanilla.py`       | Vanilla model training     |
| `eval_vanilla.py`        | Vanilla model evaluation   |
| `train_attention.py`     | Attention model training   |
| `eval_attention.py`      | Attention model evaluation |
| `attention.py`           | Attention mechanism        |
| `predictions_vanilla/`   | Vanilla predictions        |
| `predictions_attention/` | Attention predictions      |
| `attention_heatmaps/`    | Attention visualizations   |
| `connectivity.html`      | Connectivity visualization |

---

# Report and Repository Links

## W&B Report

https://wandb.ai/shreyadhondi-indian-institute-of-technology-madras/da24m019-dl-a3/reports/Shreya-Dhondi---VmlldzoxNjY5OTM2OQ

---

## GitHub Repository

https://github.com/Shreyadhondi/da24m019-dl-a3
