# Autism Trait Detection in Children using AI

This project utilizes a deep learning model built with TensorFlow to detect autism traits in children based on a dataset of behavioral questions. The model predicts whether a child exhibits autistic traits based on the responses to these questions, and it is capable of being converted into a TensorFlow Lite model for deployment on mobile and embedded devices.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [TensorFlow Lite Conversion](#tensorflow-lite-conversion)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication and behavior. Early detection can lead to better outcomes for children who may have ASD. This project leverages machine learning to detect autism traits based on answers to a series of behavioral questions.

The model, built using TensorFlow, is trained on a dataset and classifies whether a child exhibits autistic traits. Additionally, the model can be converted into TensorFlow Lite format for efficient use on mobile devices and embedded systems.

## Dataset

The dataset consists of 10 behavioral questions (Q1-Q10) and a target label (`autism`), which indicates whether the child exhibits autism traits (`Yes` or `No`).

- **Number of Samples:** 1,054
- **Features:**
  - Q1 to Q10: Responses to behavioral questions (binary integer values).
  - **Target (autism):** A binary classification of whether the child exhibits autism traits (`Yes` or `No`).

### Example Data

| Q1  | Q2  | Q3  | Q4  | Q5  | Q6  | Q7  | Q8  | Q9  | Q10 | Autism |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ |
| 0   | 0   | 0   | 0   | 0   | 0   | 1   | 1   | 0   | 1   | No     |
| 1   | 1   | 0   | 0   | 0   | 1   | 1   | 0   | 0   | 0   | Yes    |
| 1   | 0   | 0   | 0   | 0   | 0   | 1   | 1   | 0   | 1   | Yes    |
| 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   | 1   | Yes    |

### Preprocessing

- **Label Encoding:** The `autism` target column is encoded to convert the labels from "Yes"/"No" to binary integers (1 for Yes, 0 for No).
- **Standardization:** Features (Q1-Q10) are standardized using `StandardScaler`.
- **Train-Test Split:** Data is split into 80% training and 20% testing.

## Model Architecture

The model is a fully connected deep neural network (DNN) implemented in TensorFlow, with the following structure:

- **Input Layer:** The model accepts 10 input features (Q1-Q10).
- **Hidden Layers:**
  - Dense layer with 64 neurons and ReLU activation.
  - Dense layer with 32 neurons and ReLU activation.
- **Output Layer:** A single neuron with a sigmoid activation function for binary classification.

### Model Definition

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

- **Loss Function:** Binary Cross-Entropy (since this is a binary classification task).
- **Optimizer:** Adam optimizer.
- **Metrics:** Accuracy.

## Installation

### Prerequisites

- Python 3.7+
- Required libraries:
  ```bash
  pip install numpy pandas scikit-learn tensorflow
  ```

### Clone the Repository

```bash
git clone https://github.com/uxlabspk/NeuroVision.git
cd NeuroVision
```

### Prepare the Dataset

Ensure the dataset file is named `dataset.csv` and placed in the root directory.

## Usage

The code is contained in a single file, `autism_trait_detection.py`, which performs all steps including preprocessing, training, evaluation, and model conversion.

1. **Run the Script:**

   To train the model, evaluate its performance, and convert it to TensorFlow Lite, simply run:

   ```bash
   python3 autism_trait_detection.py
   ```

2. **Prediction Example:**

   The script also provides an example for predicting autism traits for a new set of data:

   ```python
   new_data = np.array([[0, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
   prediction = model.predict(new_data)
   if prediction[0][0] >= 0.5:
       print("Prediction for Autism: Yes")
   else:
       print("Prediction for Autism: No")
   ```

## TensorFlow Lite Conversion

The trained model is converted to TensorFlow Lite format for efficient deployment in Android eco system:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

This will generate a `model.tflite` file, which can be integrated in Android eco system.

## Contributing

Contributions are welcome! If you'd like to contribute to this project:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
