# 🚌 BUS_Prediction

This project predicts bus status using various deep learning models, including LSTM, GRU, Transformer, and a custom BERT-based model.

## 📋 Project Overview

The project uses historical bus status data to predict future bus statuses. Each model is trained on sequences of bus status observations and learns to predict the next status in the sequence.

## 📁 Project Structure

- **`train_comparison.py`**: 🏋️ Main training script that processes a single Excel file (`dataset/Status_100.xlsx`), trains all models sequentially, and saves individual model results to Excel files.
- **`plot_model_history.py`**: 📊 Visualization script to generate performance plots from the result Excel files.
- **`dataset/`**: 📂 Directory containing input data (Excel files).
- **`result_*.xlsx`**: 📈 Individual Excel files containing epoch-by-epoch performance metrics for each model (e.g., `result_LSTM.xlsx`, `result_GRU.xlsx`).
- **`requirements.txt`**: 📦 Python dependencies.

## ✨ Key Features

- **🎯 Single File Input**: Processes one Excel file at a time (configurable in the script).
- **📅 Time-Based Splitting**: Training and test data are split based on dates (80/20 split).
- **🔢 Sequence Padding**: Sequences shorter than the fixed length (10) are padded with -1.
- **🔄 Cumulative Prediction**: 
    - First bus predicts second bus 🚌 → 🚌
    - First + Second buses predict third bus 🚌🚌 → 🚌
    - And so on...
- **📄 Individual Model Results**: Each model's training history is saved to a separate Excel file.

## 🌟 Environment Setup

This project uses the Conda environment named **"Normal"**.

1. **Ensure Conda is installed.** 🐍
2. **Activate the environment:**
    ```bash
    conda activate Normal
    ```

## 📦 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. 🏋️ Train Models

To train all models (LSTM, GRU, Transformer, BERT) on the configured dataset:

```bash
conda run -n Normal python train_comparison.py
```

**What happens:**
- 📥 Loads data from `dataset/Status_100.xlsx`.
- ✂️ Splits data by date (first 80% for training, last 20% for testing).
- 🎓 Trains each model for the configured number of epochs.
- 💾 Saves model checkpoints (e.g., `bus_lstm_model.pth`).
- 📊 Saves training history to individual Excel files (e.g., `result_LSTM.xlsx`).

### 2. 📊 Visualize Results

To generate performance comparison plots from the result files:

```bash
conda run -n Normal python plot_model_history.py
```

This creates `model_comparison_plot.png` showing Train/Test Loss and Accuracy for all models across epochs. 📈

## ⚙️ Configuration

You can modify the following parameters in `train_comparison.py`:

- **`DATA_FILE`**: 📂 Path to the input Excel file (default: `'./dataset/Status_100.xlsx'`)
- **`SEQUENCE_LENGTH`**: 🔢 Fixed sequence length with padding (default: `10`)
- **`BATCH_SIZE`**: 📦 Training batch size (default: `32`)
- **`HIDDEN_SIZE`**: 🧠 Hidden layer size for LSTM/GRU/BERT (default: `256`)
- **`NUM_LAYERS`**: 🏗️ Number of layers (default: `3`)
- **`LEARNING_RATE`**: 📉 Learning rate (default: `0.001`)
- **`NUM_EPOCHS`**: 🔄 Number of training epochs (default: `50`)
- **`TRAIN_SPLIT_RATIO`**: ✂️ Ratio for train/test split by date (default: `0.8`)

## 🤖 Models Implemented

- **LSTM** (Long Short-Term Memory) 🧠
- **GRU** (Gated Recurrent Unit) 🔄
- **Transformer** (Encoder-only with positional encoding) 🤖
- **BERT** (Custom implementation using Hugging Face configuration) 📚
