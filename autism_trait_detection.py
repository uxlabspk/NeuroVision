import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Load Data
data = pd.read_csv('dataset.csv')

# Data Preprocessing
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target variable

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Shuffle and Split Data
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Prediction
# Example: Predicting on new data
new_data = np.array([[0, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
new_data = scaler.transform(new_data)  # Standardize new data
prediction = model.predict(new_data)
if prediction[0][0] >= 0.5:
    print("Prediction for Autism in case: Yes")
else:
    print("Prediction for Autism in case: No")

# Save the trained model
# model.save('logistic_regression_model_tf.tflite')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Step 3: Save the TensorFlow Lite model to a .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)



