import pandas as pd
import os

class DataLoader:
    PROJECT_FOLDER = "AI_Project"
    DATA_SUBDIR = "data"

    FEATURE_NAMES = [
        "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
        "predator", "toothed", "backbone", "breathes", "venomous", 
        "fins", "legs", "tail", "domestic", "catsize"
    ]

    CLASS_NAMES = {
        1: "Mammal (Thú)", 2: "Bird (Chim)", 3: "Reptile (Bò sát)",
        4: "Fish (Cá)", 5: "Amphibian (Lưỡng cư)", 6: "Bug (Côn trùng)",
        7: "Invertebrate (Động vật không xương sống)"
    }
    def __init__(self):
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        self.DATA_DIR = os.path.join(current_dir, "..", "data")
        self.DATA_DIR = os.path.normpath(self.DATA_DIR)

    def load_data(self):
     
        file_path_2 = os.path.join(self.DATA_DIR, "zoo2.csv")
        file_path_3 = os.path.join(self.DATA_DIR, "zoo3.csv")

        try:
            df2 = pd.read_csv(file_path_2)
            df3 = pd.read_csv(file_path_3)
            full_df = pd.concat([df2, df3], ignore_index=True)

            full_df.columns = ["animal_name"] + self.FEATURE_NAMES + ["class_type"]

            X = full_df[self.FEATURE_NAMES]
            y = full_df["class_type"] - 1

            print(f"Đã load tổng cộng {len(full_df)} mẫu dữ liệu.")
            return X, y

        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file dữ liệu. {e}")
            return None, None