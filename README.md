<div align="center">

# NeuroVision

### Detecting autism traits in children using AI.

A **lightweight deep learning model** that classifies autism traits from 10 behavioral questions. Trained in seconds. Deployable on mobile via TensorFlow Lite.

Built with TensorFlow and scikit-learn. One script. One dataset. No cloud required.

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-StandardScaler-FF6B35?style=flat-square)](https://scikit-learn.org/)
[![TFLite](https://img.shields.io/badge/TFLite-Mobile-4285F4?style=flat-square)](https://www.tensorflow.org/lite)
[![License](https://img.shields.io/badge/License-MIT-00C853?style=flat-square)](LICENSE)

</div>

---

## Why NeuroVision?

Autism Spectrum Disorder affects communication and behavior. Early detection improves outcomes, but screening tools are often expensive or inaccessible. NeuroVision provides a fast, local, and privacy-respecting model that classifies autism traits from simple behavioral questionnaires — no internet, no API keys, no data leaves your machine.

> "Early detection shouldn't require a hospital visit."

---

## Features

### One-Script Pipeline

Preprocessing, training, evaluation, and TFLite conversion all happen in a single `autism_trait_detection.py`. Run it once and you're done.

### Mobile-Ready

The trained model converts to TensorFlow Lite automatically, producing a `model.tflite` file ready for Android and embedded deployment.

### No Dependencies Beyond the Basics

TensorFlow, scikit-learn, pandas, numpy. No heavy frameworks, no GPU requirement. Trains in under a minute on CPU.

### Configurable Architecture

Swap the hidden layer sizes, epochs, or batch size in a few lines. The model is a straightforward Sequential DNN — easy to understand and modify.

### And more

- **Binary classification** — predicts Yes/No autism traits
- **StandardScaler** — features standardized for consistent predictions
- **Shuffled split** — 80/20 train-test with reproducible seed
- **Label encoding** — handles string labels automatically
- **Lightweight output** — small `.tflite` file suitable for on-device inference

---

## Quick Start

### Prerequisites

- Python 3.7+
- TensorFlow, scikit-learn, pandas, numpy

### Install

```bash
git clone https://github.com/uxlabspk/NeuroVision.git
cd NeuroVision
pip install numpy pandas scikit-learn tensorflow
```

### Run

```bash
python3 autism_trait_detection.py
```

This will train the model, print accuracy, run a sample prediction, and save `model.tflite`.

### Predict on New Data

```python
import numpy as np
new_data = np.array([[0, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
if prediction[0][0] >= 0.5:
    print("Prediction for Autism: Yes")
else:
    print("Prediction for Autism: No")
```

---

## How it works

```
dataset.csv
    ↓
LabelEncoder          Encodes string columns (Yes/No) to integers
    ↓
StandardScaler        Normalizes features to zero mean, unit variance
    ↓
train_test_split      80% train, 20% test (shuffled)
    ↓
DNN (64 → 32 → 1)    Binary classifier with ReLU hidden layers
    ↓
model.evaluate        Reports test accuracy
    ↓
TFLiteConverter       Converts Keras model to .tflite for mobile
```

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Model | **TensorFlow Sequential DNN** — 64 → 32 → 1 (sigmoid) |
| Preprocessing | **scikit-learn** — LabelEncoder, StandardScaler, train_test_split |
| Data | **pandas** — CSV loading and manipulation |
| Deployment | **TFLite** — mobile and embedded inference |

---

## Project Structure

```
NeuroVision/
├── autism_trait_detection.py   Full pipeline: train, evaluate, convert
├── dataset.csv                 Behavioral questionnaire data (1054 samples)
├── model.tflite                Pre-converted TFLite model
├── LICENSE                     MIT
└── README.md
```

---

## Contributing

Contributions welcome.

1. Fork it
2. Create a branch (`git checkout -b feat/my-thing`)
3. Commit (`git commit -m 'Add my thing'`)
4. Push (`git push origin feat/my-thing`)
5. Open a PR

---

## License

MIT — do whatever you want with it.

---

**If NeuroVision helps your project, give it a star.**

It helps others find it, and tells me this is worth continuing.

[Star this repo](https://github.com/uxlabspk/NeuroVision/stargazers)
