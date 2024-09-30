# Import Libraries & Packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import warnings
# Ignore warnings
warnings.filterwarnings('ignore')

# Load Data
data = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')

# Show basic data information
print(data.head())
print("============================")
print(data.shape)
print("============================")
print(data.info())
print("============================")
print(data.describe())
print("============================")
print(data.duplicated().sum())
print("============================")

# Dropping Unneeded Columns
data.drop(columns=['id'], inplace=True)
print(data.head())
print("============================")

# Define Features 'X' and Target 'y'
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(X)
print("============================")
print(y)
print("============================")

# Data Encoding
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.values.reshape(-1, 1))
print(y)
print("============================")

# Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Data Splitting
X_train, X_dummy, y_train, y_dummy = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_dummy, y_dummy, test_size=0.5, random_state=42)
print(X_train.shape)
print("============================")
print(X_val.shape)
print("============================")
print(X_test.shape)
print("============================")

# Deep Learning Model
model = Sequential([
    Dense(1024, activation='relu', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model Training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[callback])

# Model Evaluation
train_eval = model.evaluate(X_train, y_train)
val_eval = model.evaluate(X_val, y_val)
test_eval = model.evaluate(X_test, y_test)

print(f"Train Accuracy: {train_eval[1]}")
print(f"Validation Accuracy: {val_eval[1]}")
print(f"Test Accuracy: {test_eval[1]}")

# Classification Report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Plotting Training History

# Accuracy Plot
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss Plot
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
