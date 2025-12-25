import streamlit as st
import pandas as pd
import numpy as np
import joblib
from dataLoader import DataLoader
from tensorflow import keras
from pathlib import Path

st.set_page_config(page_title="Phân loại động vật", page_icon="🦁")
st.title("🦁 Dự đoán lớp động vật (Zoo Classification)")

@st.cache_resource
def load_models():
    model_dir = Path("data/model")
    try:
        return {
            "Decision Tree": joblib.load(model_dir / "decision_tree.pkl"),
            "Random Forest": joblib.load(model_dir / "random_forest.pkl"),
            "KNN": joblib.load(model_dir / "knn.pkl"),
            "TabTransformer": keras.models.load_model(model_dir / "tabtransformer.keras")
        }
    except FileNotFoundError:
        return None

models = load_models()

st.sidebar.header("Chọn đặc điểm")
input_data = {}

for feature in DataLoader.FEATURE_NAMES:
    if feature == "legs":
        input_data[feature] = st.sidebar.slider("Số chân (Legs)", 0, 8, 4)
    else:
        label = f"{feature.capitalize()}?" 
        input_data[feature] = st.sidebar.selectbox(label, [0, 1])

input_df = pd.DataFrame([input_data])

st.write("Đặc điểm đã chọn:", input_df)

if st.button("Dự đoán ngay"):
    if not models:
        st.error("Chưa tìm thấy model. Hãy chạy trainModel.py trước!")
    else:
        cols = st.columns(len(models))
        X_input = input_df.values
        
        for idx, (name, model) in enumerate(models.items()):
            with cols[idx]:
                if name == "TabTransformer":
                    pred = np.argmax(model.predict(X_input), axis=1)[0]
                else:
                    pred = model.predict(X_input)[0]
                
                class_name = DataLoader.CLASS_NAMES.get(pred + 1, "Unknown")
                
                st.info(f"**{name}**")
                st.success(class_name)