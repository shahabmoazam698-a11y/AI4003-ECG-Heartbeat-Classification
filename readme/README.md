# ECG Heartbeat Classification Using a Neural Network

## Project Overview

This project classifies ECG heartbeat signals into five categories using a manually implemented neural network built with PyTorch.

The model uses a fully connected architecture with:
- Input layer: 187 features
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 5 classes

The network uses manually implemented forward propagation, cross-entropy loss, and momentum-based gradient updates.

## Dataset

We use the MIT-BIH Arrhythmia heartbeat dataset from Kaggle.

Dataset link: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Required files:
- `mitbih_train.csv`
- `mitbih_test.csv`

Place both files inside the `data/` folder.

## Heartbeat Classes

| Label | Class |
|---|---|
| 0 | N — Normal |
| 1 | S — Supraventricular ectopic |
| 2 | V — Ventricular ectopic |
| 3 | F — Fusion |
| 4 | Q — Unknown / Paced |

## Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 97.85% |
| Balanced Accuracy | 82.13% |
| Majority-Class Baseline | 82.76% |

## Project Structure

```text
AI4003_Assignment03/
│
├── data/
│   ├── mitbih_train.csv
│   └── mitbih_test.csv
│
├── src/
│   ├── load_data.py
│   ├── prepare.py
│   ├── plot_beats.py
│   ├── model.py
│   ├── loss.py
│   ├── grad_check.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── best_model.pth
│
├── app/
│   └── app/
│       └── app.py
│
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install torch numpy pandas matplotlib streamlit
```

## How to Run

Run gradient checking:

```bash
python src/grad_check.py
```

Train the model:

```bash
python src/train.py
```

Evaluate the model:

```bash
python src/evaluate.py
```

Launch the Streamlit application:

```bash
streamlit run app/app/app.py
```

Open the local URL shown in the terminal to use the application.

## Application Features

- Select ECG heartbeat samples.
- Visualize ECG waveforms.
- Predict heartbeat categories.
- View prediction probabilities.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Academic Project

Course: AI4003 — Applied Machine Learning

Project: ECG Heartbeat Classification

Group: Two Students
