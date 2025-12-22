import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from pathlib import Path

st.set_page_config(page_title="Phân loại động vật", page_icon="🦁")
st.title("🦁 Dự đoán lớp động vật (Zoo Classification)")
st.write("Nhập các đặc điểm bên dưới để phân loại động vật.")

@st.cache_resource
def load_models():
    model_dir = Path(__file__).resolve().parent.parent / "data" / "model"
    
    try:
        models = {
            "Decision Tree": joblib.load(model_dir / "decision_tree.pkl"),
            "Random Forest": joblib.load(model_dir / "random_forest.pkl"),
            "KNN": joblib.load(model_dir / "knn.pkl"),
            "TabTransformer": keras.models.load_model(model_dir / "tabtransformer.keras")
        }
        return models
    except FileNotFoundError:
        st.error("Chưa tìm thấy file model! Vui lòng chạy file trainModel.py trước.")
        return None

models = load_models()

class_names = {
    1: "Mammal (Thú)",
    2: "Bird (Chim)",
    3: "Reptile (Bò sát)",
    4: "Fish (Cá)",
    5: "Amphibian (Lưỡng cư)",
    6: "Bug (Côn trùng)",
    7: "Invertebrate (Động vật không xương sống)"
}

st.sidebar.header("Chọn đặc điểm")

def user_input_features():
    hair = st.sidebar.selectbox("Có lông tóc (Hair)?", [0, 1])
    feathers = st.sidebar.selectbox("Có lông vũ (Feathers)?", [0, 1])
    eggs = st.sidebar.selectbox("Đẻ trứng (Eggs)?", [0, 1])
    milk = st.sidebar.selectbox("Có sữa (Milk)?", [0, 1])
    airborne = st.sidebar.selectbox("Biết bay (Airborne)?", [0, 1])
    aquatic = st.sidebar.selectbox("Sống dưới nước (Aquatic)?", [0, 1])
    predator = st.sidebar.selectbox("Săn mồi (Predator)?", [0, 1])
    toothed = st.sidebar.selectbox("Có răng (Toothed)?", [0, 1])
    backbone = st.sidebar.selectbox("Có xương sống (Backbone)?", [0, 1])
    breathes = st.sidebar.selectbox("Thở bằng phổi (Breathes)?", [0, 1])
    venomous = st.sidebar.selectbox("Có độc (Venomous)?", [0, 1])
    fins = st.sidebar.selectbox("Có vây (Fins)?", [0, 1])
    legs = st.sidebar.slider("Số chân (Legs)", 0, 8, 4) 
    tail = st.sidebar.selectbox("Có đuôi (Tail)?", [0, 1])
    domestic = st.sidebar.selectbox("Được thuần hóa (Domestic)?", [0, 1])
    catsize = st.sidebar.selectbox("Kích thước bằng mèo (Catsize)?", [0, 1])

    data = {
        'hair': hair, 'feathers': feathers, 'eggs': eggs, 'milk': milk,
        'airborne': airborne, 'aquatic': aquatic, 'predator': predator, 'toothed': toothed,
        'backbone': backbone, 'breathes': breathes, 'venomous': venomous, 'fins': fins,
        'legs': legs, 'tail': tail, 'domestic': domestic, 'catsize': catsize
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Đặc điểm bạn đã chọn:")
st.write(input_df)

if st.button("Dự đoán ngay"):
    if models:
        st.subheader("Kết quả dự đoán:")
        
        X_input = input_df.values

        cols = st.columns(len(models))
        
        for idx, (name, model) in enumerate(models.items()):
            with cols[idx]:
                if name == "TabTransformer":
                    y_prob = model.predict(X_input)
                    prediction = np.argmax(y_prob, axis=1)[0] + 1
                else:
                    prediction = model.predict(X_input)[0] + 1 

                st.info(f"**{name}**")
                st.success(f"{class_names.get(prediction, 'Unknown')}")