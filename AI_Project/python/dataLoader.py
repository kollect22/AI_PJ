import pandas as pd
import os

class DataLoader:
    PROJECT_FOLDER = "AI_Project"
    DATA_SUBDIR = "data"
    DATA_DIR = os.path.join(os.getcwd(), PROJECT_FOLDER, DATA_SUBDIR)

    def load_data_frame(self):
     
        file_path_2 = os.path.join(self.DATA_DIR, "zoo2.csv")
        file_path_3 = os.path.join(self.DATA_DIR, "zoo3.csv")

        try:
            df2 = pd.read_csv(file_path_2)
            df3 = pd.read_csv(file_path_3)

            full_df = pd.concat([df2, df3], ignore_index=True)

            print(f"Đã load tổng cộng {len(full_df)} mẫu dữ liệu.")
            return full_df

        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file dữ liệu. {e}")
            return None

    def get_features_and_labels(self, df):
       
        if df is None: return None, None

        X = df.iloc[:, 1:17].values 

        y = df.iloc[:, 17].values   

        return X, y

if __name__ == "__main__":
    loader = DataLoader()
    
    df = loader.load_data_frame()
    
    if df is not None:
        print("Dữ liệu 5 dòng đầu:")
        print(df.head())


        X, y = loader.get_features_and_labels(df)
        
        print("\nKích thước tập features (X):", X.shape)
        print("Kích thước tập nhãn (y):", y.shape)
        print("\nFeatures mẫu đầu tiên:", X[0])