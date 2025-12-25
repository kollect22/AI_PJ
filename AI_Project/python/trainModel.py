import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import joblib
from tensorflow.keras import layers, models, optimizers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from sklearn.metrics import confusion_matrix    
from dataLoader import DataLoader

loader = DataLoader()
X, y = loader.load_data()

# chia train - test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

model_dir = Path(loader.DATA_DIR) / "model"
model_dir.mkdir(parents=True, exist_ok=True)

models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

print("-" * 65)
print(f"{'Model':<15} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = calc_metrics(y_test, y_pred)
    print(f"{name:<15} | {metrics[0]:.4f}   | {metrics[1]:.4f}   | {metrics[2]:.4f}   | {metrics[3]:.4f}")
    
    joblib.dump(model, model_dir / f"{name.lower()}.pkl")

def build_simple_tabtransformer(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    
    projection = layers.Dense(64, activation="relu")(inputs)
    x = layers.Reshape((4, 16))(projection) 
    
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization(epsilon = 1e-6)(x)

    #feed forward network
    x_ffn = layers.Dense(16, activation="relu")(x)
    x_ffn = layers.Dense(16)(x_ffn)
    x = layers.Add()([x, x_ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

tab_model = build_simple_tabtransformer(input_dim=16, num_classes=7)

tab_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

print("\nĐang train TabTransformer...")
history = tab_model.fit(
    X_train, y_train, 
    epochs=70,
    batch_size=16, 
    validation_data=(X_test, y_test),
    verbose=0
)

y_prob = tab_model.predict(X_test)
tt_y_pred = np.argmax(y_prob, axis=1)
tt_metrics = calc_metrics(y_test, tt_y_pred)

print(f"{'TabTransformer':<15} | {tt_metrics[0]:.4f}   | {tt_metrics[1]:.4f}   | {tt_metrics[2]:.4f}   | {tt_metrics[3]:.4f}")
print("-" * 65)

tab_model.save(model_dir / "tabtransformer.keras")

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1,8), yticklabels=range(1,8))
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(model_dir / filename)
    plt.close()

plot_cm(y_test, tt_y_pred, "Confusion Matrix - TabTransformer", "cm_tabtransformer.png")

print(f"Đã hoàn tất! Model được lưu tại: {model_dir}")