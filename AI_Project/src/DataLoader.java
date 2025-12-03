import java.io.*;
import java.util.*;

public class Dataloader {

    // Đọc dữ liệu từ CSV và trả về danh sách Animal
    public List<Animal> loadData(String filePath) throws IOException {
        List<Animal> animals = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));

        String line;
        br.readLine();

        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");

            if (parts.length >= 17) {
                int[] features = new int[16];


                for (int i = 0; i < 16; i++) {
                    features[i] = Integer.parseInt(parts[i + 1].trim());
                }

                int type = Integer.parseInt(parts[17].trim());
                animals.add(new Animal(features, type));
            }
        }

        br.close();
        return animals;
    }

    // Chia dữ liệu thành tập huấn luyện và kiểm tra
    public Object[] splitData(List<Animal> animals, double trainRatio) {
        Collections.shuffle(animals); // Trộn ngẫu nhiên

        int trainSize = (int)(animals.size() * trainRatio);

        double[][] trainFeatures = new double[trainSize][16];
        int[] trainLabels = new int[trainSize];

        double[][] testFeatures = new double[animals.size() - trainSize][16];
        int[] testLabels = new int[animals.size() - trainSize];

        // huấn luyện
        for (int i = 0; i < trainSize; i++) {
            trainFeatures[i] = animals.get(i).getFeatures();
            trainLabels[i] = animals.get(i).getType();
        }


        for (int i = trainSize; i < animals.size(); i++) {
            testFeatures[i - trainSize] = animals.get(i).getFeatures();
            testLabels[i - trainSize] = animals.get(i).getType();
        }

        return new Object[]{trainFeatures, trainLabels, testFeatures, testLabels};
    }
}