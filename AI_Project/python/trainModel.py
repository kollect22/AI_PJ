import numpy as np
import pandas as pd
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

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_y_pred = dt.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

def calc_metrics(y_pred, y_test):
    return accuracy_score(y_pred, y_test), precision_score(y_pred, y_test, average='weighted'), recall_score(y_pred, y_test, average='weighted'), f1_score(y_pred, y_test, average='weighted')  

print(calc_metrics(dt_y_pred, y_test))
print(calc_metrics(rf_y_pred, y_test))


# lưu các model đã train vào data/model
model_dir = Path(__file__).resolve().parent.parent / "data" / "model"
model_dir.mkdir(exist_ok=True)

joblib.dump(dt, model_dir / "decision_tree.pkl")
joblib.dump(rf, model_dir / "random_forest.pkl")
joblib.dump(knn, model_dir / "knn.pkl")