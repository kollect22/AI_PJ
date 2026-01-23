import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pytorch_tabular import TabularModel
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
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

X = X.astype(float)
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
    "DecisionTree": DecisionTreeClassifier(criterion='entropy', random_state= 42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
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

#tabtran
print("-" * 30)
print("Đang huấn luyện TabTransformer...")

df = X.copy()
df["label"] = y 

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

target_col_name = "label" 

continuous_cols = ['legs']
categorical_cols = [col for col in X.columns if col != 'legs']

data_config = DataConfig(
    target=[target_col_name],
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    num_workers=0
)

model_config = TabTransformerConfig(
    task="classification",
    metrics=["accuracy"],
    input_embed_dim=16, 
    num_heads=4, 
    num_attn_blocks=2
)

trainer_config = TrainerConfig(
    max_epochs=70,
    batch_size=16,
    accelerator="cpu",
    trainer_kwargs={
        "enable_progress_bar": False, 
        "log_every_n_steps": 1
    }
)

optimizer_config = OptimizerConfig(optimizer="Adam")

tab_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

tab_model.fit(train=train_df, validation=test_df)

pred_df = tab_model.predict(test_df)

y_true_tt = test_df[target_col_name].values
y_pred_tt = pred_df["label_prediction"].values

tt_metrics = calc_metrics(y_true_tt, y_pred_tt)

print(f"{'TabTransformer':<15} | {tt_metrics[0]:.4f}   | {tt_metrics[1]:.4f}   | {tt_metrics[2]:.4f}   | {tt_metrics[3]:.4f}")
print("-" * 65)

tab_model.save_model(model_dir / "tabtransformer_pytorch")

#confusion matrix
def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(0,7), yticklabels=range(0,7))
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(model_dir / filename)
    plt.close()

plot_cm(y_test, y_pred_tt, "Confusion Matrix - TabTransformer", "cm_tabtransformer.png")

print(f"Đã hoàn tất! Model được lưu tại: {model_dir}")