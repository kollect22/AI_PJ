package src.main;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.out.println("--- BẮT ĐẦU HỆ THỐNG PHÂN LOẠI ĐỘNG VẬT ---");

        // 1. Tải dữ liệu (Chỉ cần cho mục đích testing/display)
        DataLoader loader = new DataLoader();
        List<Animal> allAnimals = loader.loadAllData();
        
        // Kiểm tra xem dữ liệu có được tải không
        if (allAnimals.isEmpty()) {
            System.err.println("Không có dữ liệu, không thể tiếp tục.");
            return;
        }

        // 2. Chuẩn bị mẫu thử
        // Lấy một con vật bất kỳ để kiểm tra
        Animal sampleAnimal = allAnimals.get(0);
        double[] features = sampleAnimal.getFeatures();
        int trueType = sampleAnimal.getType();

        // 3. Khởi tạo và gọi Python Bridge
        PythonBridge bridge = new PythonBridge();
        
        System.out.println("------------------------------------");
        System.out.println("Kiểm tra kết nối với Python Model:");
        System.out.println("Đặc trưng mẫu: " + features.length + " thuộc tính.");
        System.out.println("Loại thực tế (True Type): " + trueType);

       testModel(bridge, features, trueType, "knn.pkl", "K-Nearest Neighbors");
        testModel(bridge, features, trueType, "decision_tree.pkl", "Decision Tree");
        testModel(bridge, features, trueType, "random_forest.pkl", "Random Forest");
    }

    /**
     * Hàm phụ trợ để gọi cầu nối Python và in kết quả.
     */
    private static void testModel(PythonBridge bridge, double[] features, int trueType, String modelFileName, String modelDisplay) {
        
        // SỬA LỖI TẠI ĐÂY: Thêm tham số modelFileName
        int predictedLabel = bridge.predictAnimalType(features, modelFileName);
        
        int finalPredictedType = predictedLabel; 
        
        System.out.print(String.format("Model %s (%s): ", modelDisplay, modelFileName));

        if (predictedLabel > 0 && predictedLabel < 8) { // Kiểm tra nhãn hợp lệ (1-7)
            System.out.println("Dự đoán Loại " + finalPredictedType);
            if (finalPredictedType == trueType) {
                System.out.println("    => DỰ ĐOÁN CHÍNH XÁC! ");
            } else {
                System.out.println("    => DỰ ĐOÁN SAI. ");
            }
        } else {
            System.err.println("GỌI PYTHON BRIDGE THẤT BẠI. Kết quả lỗi: " + predictedLabel);
        }
    }
}