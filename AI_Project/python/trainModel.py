import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path    
import joblib

# lấy path & đọc các dataset
csv_path1 = Path(__file__).resolve().parent.parent / 'data' / 'zoo2.csv'
csv_path2 = Path(__file__).resolve().parent.parent / 'data' / 'zoo3.csv'

df1 = pd.read_csv(csv_path1)
df2 = pd.read_csv(csv_path2)

df= pd.concat([df1, df2], ignore_index=True)

df.columns = [
    "animal_name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous", "fins",
    "legs", "tail", "domestic", "catsize", "class_type"
]

X = df.drop(columns=['class_type', 'animal_name'])
y = df['class_type'] - 1

# chia train - test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

def calc_metrics(y_pred, y_test):
    return accuracy_score(y_pred, y_test), precision_score(y_pred, y_test, average='weighted'), recall_score(y_pred, y_test, average='weighted'), f1_score(y_pred, y_test, average='weighted')  

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_y_pred = dt.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

def build_simple_tabtransformer(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Reshape((1, 32))(x) 
    
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

tab_model = build_simple_tabtransformer(input_dim=16, num_classes=7)

tab_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

tab_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

y_prob = tab_model.predict(X_test)
tt_y_pred = np.argmax(y_prob, axis=1)

print("TabTransformer Results:", calc_metrics(tt_y_pred, y_test))
print("DecisionTree Results: ", calc_metrics(dt_y_pred, y_test))
print("RandomForest Results: ",calc_metrics(rf_y_pred, y_test))
print("KNN Results: ",calc_metrics(knn_y_pred, y_test))


# lưu các model đã train
model_dir = Path(__file__).resolve().parent.parent / "data" / "model"
model_dir.mkdir(exist_ok=True)

joblib.dump(dt, model_dir / "decision_tree.pkl")
joblib.dump(rf, model_dir / "random_forest.pkl")
joblib.dump(knn, model_dir / "knn.pkl")
tab_model.save(model_dir / "tabtransformer.keras")