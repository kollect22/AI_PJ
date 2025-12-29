import streamlit as st
import pandas as pd
import numpy as np
import joblib
from dataLoader import DataLoader
from pytorch_tabular import TabularModel
from pathlib import Path

st.set_page_config(page_title="Phân loại động vật", page_icon="🦁")
st.title("🦁 Dự đoán lớp động vật (Zoo Classification)")

loader = DataLoader()
model_dir = Path(loader.DATA_DIR) / "model"

@st.cache_resource
def load_models():
    models = {}
    try:
        models["Decision Tree"] = joblib.load(model_dir / "decisiontree.pkl")
        models["Random Forest"] = joblib.load(model_dir / "randomforest.pkl")
        models["KNN"] = joblib.load(model_dir / "knn.pkl")
 
        tab_path = model_dir / "tabtransformer_pytorch"
        models["TabTransformer"] = TabularModel.load_model(tab_path)

        return models
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
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
        
        for idx, (name, model) in enumerate(models.items()):
            with cols[idx]:
                if name == "TabTransformer":
                    pred_df = model.predict(input_df)
                    pred = pred_df["label_prediction"].values[0]
                else:
                    pred = model.predict(input_df.values)[0]
                
                class_name = DataLoader.CLASS_NAMES.get(pred + 1, "Unknown")
                
                st.info(f"**{name}**")
                st.success(class_name)