import java.io.*;
import java.util.*;

public class ModelTrainer {
    private int k = 3; // Mặc định 3 neighbors
    private double[][] trainData;
    private int[] trainLabels;

    // Huấn luyện = chỉ lưu dữ liệu
    public void train(double[][] trainData, int[] trainLabels) {
        this.trainData = trainData;
        this.trainLabels = trainLabels;
        System.out.println("Đã lưu " + trainData.length + " mẫu huấn luyện");
    }

    // Dự đoán cho 1 mẫu
    public int predict(double[] sample) {
        // Tính khoảng cách đến tất cả mẫu huấn luyện
        List<Neighbor> neighbors = new ArrayList<>();

        for (int i = 0; i < trainData.length; i++) {
            double distance = 0;
            for (int j = 0; j < sample.length; j++) {
                double diff = sample[j] - trainData[i][j];
                distance += diff * diff;
            }
            distance = Math.sqrt(distance);
            neighbors.add(new Neighbor(distance, trainLabels[i]));
        }

        // Sắp xếp theo khoảng cách
        neighbors.sort(Comparator.comparingDouble(n -> n.distance));

        // Đếm phiếu của k neighbors gần nhất
        Map<Integer, Integer> votes = new HashMap<>();
        for (int i = 0; i < Math.min(k, neighbors.size()); i++) {
            int label = neighbors.get(i).label;
            votes.put(label, votes.getOrDefault(label, 0) + 1);
        }

        // Tìm label có nhiều phiếu nhất
        int bestLabel = -1;
        int maxVotes = 0;
        for (Map.Entry<Integer, Integer> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                bestLabel = entry.getKey();
            }
        }

        return bestLabel;
    }

    // Đánh giá độ chính xác
    public double evaluate(double[][] testData, int[] testLabels) {
        int correct = 0;
        for (int i = 0; i < testData.length; i++) {
            if (predict(testData[i]) == testLabels[i]) {
                correct++;
            }
        }
        return (double)correct / testData.length;
    }

    // Lớp phụ trợ
    private static class Neighbor {
        double distance;
        int label;

        Neighbor(double distance, int label) {
            this.distance = distance;
            this.label = label;
        }
    }

    // Getter/Setter
    public void setK(int k) { this.k = k; }
    public int getK() { return k; }

    // Lưu và tải mô hình
    public void save(String filePath) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(
                new FileOutputStream(filePath))) {
            out.writeObject(this);
        }
    }

    public static ModelTrainer load(String filePath)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(
                new FileInputStream(filePath))) {
            return (ModelTrainer) in.readObject();
        }
    }
}