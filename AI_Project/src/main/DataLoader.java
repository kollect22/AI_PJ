package src.main;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class DataLoader {

    private static final String PROJECT_FOLDER = "AI_Project"; 
    private static final String DATA_SUBDIR = "data";
    private static final String DATA_DIR = Paths.get(PROJECT_FOLDER, DATA_SUBDIR).toAbsolutePath().toString();

    public List<Animal> loadData(String filename) {
        List<Animal> animals = new ArrayList<>();
        Path filePath = Paths.get(DATA_DIR, filename);

        try (BufferedReader br = new BufferedReader(new FileReader(filePath.toFile()))) {
            String line;
            boolean isHeader = true;
            
            while ((line = br.readLine()) != null) {
                if (isHeader) {
                    isHeader = false; // Bỏ qua dòng header
                    continue;
                }
                
                String[] values = line.split(",");
                // Bỏ qua cột đầu tiên (animal_name)
                
                // Lấy 16 đặc trưng (hair, feathers, ...)
                int[] features = new int[16]; 
                for (int i = 0; i < 16; i++) {
                    // Dữ liệu bắt đầu từ cột thứ 2 (index 1) đến cột 17 (index 16)
                    features[i] = Integer.parseInt(values[i + 1].trim()); 
                }
                
                // Cột cuối cùng là class_type (index 17)
                int classType = Integer.parseInt(values[17].trim()); 
                
                animals.add(new Animal(features, classType));
            }
        } catch (IOException e) {
            System.err.println("Lỗi đọc file: " + filePath.toString() + ". Chi tiết: " + e.getMessage());
        } catch (NumberFormatException e) {
            System.err.println("Lỗi định dạng số trong file CSV: " + e.getMessage());
        }
        return animals;
    }

    public List<Animal> loadAllData() {
        List<Animal> allAnimals = loadData("zoo2.csv");
        allAnimals.addAll(loadData("zoo3.csv"));
        System.out.println("Đã load tổng cộng " + allAnimals.size() + " mẫu động vật.");
        return allAnimals;
    }
}