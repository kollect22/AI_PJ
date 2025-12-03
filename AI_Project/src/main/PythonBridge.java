package src.main; 

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.nio.file.Path;

/**
 * Lớp này chịu trách nhiệm gọi script Python để sử dụng các mô hình ML 
 * (KNN, DT, RF) đã được huấn luyện và lưu bằng joblib.
 */
public class PythonBridge {
    private static final String PROJECT_FOLDER = "AI_Project"; 
    private static final String SCRIPT_DIR = "python";
    private static final String PYTHON_SCRIPT_NAME = "predict.py";
    // Đặt đường dẫn tuyệt đối đến thư mục chứa script Python
    private static final String PYTHON_SCRIPT_PATH = Paths.get(PROJECT_FOLDER, SCRIPT_DIR, PYTHON_SCRIPT_NAME).toAbsolutePath().toString();
    
    /**
     * Phương thức gọi script Python để dự đoán loại động vật.
     * @param features Mảng double chứa 16 đặc trưng (hair, feathers, ...)
     * @param modelName Tên file model muốn dùng 
     * @return Loại động vật dự đoán (1-7) hoặc -1 nếu có lỗi.
     */
    public int predictAnimalType(double[] features, String modelName) {
        
        // 1. Chuyển mảng double[] thành chuỗi "1,0,0,1..." để Python đọc
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < features.length; i++) {
            // Ép kiểu double về int 
            sb.append((int) features[i]); 
            if (i < features.length - 1) {
                sb.append(",");
            }
        }
        String featureString = sb.toString();

        try {
            // 2. Cấu hình lệnh gọi Python 
            // Lệnh sẽ chạy: python predict.py "1,0,0,1,..." knn.pkl
            ProcessBuilder pb = new ProcessBuilder(
                "python",        
                PYTHON_SCRIPT_PATH, // Đường dẫn tới file script Python
                featureString,      // Tham số 1: Data
                modelName           // Tham số 2: Tên Model
            );
            
            // Tùy chọn: Chuyển hướng lỗi của Python vào luồng lỗi chuẩn của Java
            // pb.redirectErrorStream(true); 
            
            Process process = pb.start();

            // 3. Đọc kết quả từ luồng output của Python
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()));
            
            String line = reader.readLine();

            // Chờ quá trình Python kết thúc
            int exitCode = process.waitFor();
            
            if (exitCode != 0) {
                System.err.println("Lỗi Python Script. Mã thoát: " + exitCode + ", Output: " + line);
                
                // Đọc thêm luồng lỗi để debug (nếu không dùng redirectErrorStream)
                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                String errorLine;
                while ((errorLine = errorReader.readLine()) != null) {
                    System.err.println("Python Error: " + errorLine);
                }
                return -1; 
            }
            
            // 4. Phân tích kết quả
            if (line != null) {
                // Kết quả prediction từ 0 đến 6 (nhãn đã trừ 1)
                int predictedLabel = Integer.parseInt(line.trim());
                // Cộng thêm 1 để trả về loại động vật (1-7)
                return predictedLabel + 1; 
            } else {
                System.err.println("Python trả về kết quả rỗng (Null).");
                return -1;
            }

        } catch (IOException | InterruptedException | NumberFormatException e) {
            System.err.println("Lỗi gọi Python hoặc xử lý dữ liệu: " + e.getMessage());
            return -1;
        }
    }
}